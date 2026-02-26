import time
import os
import glob
import threading
from datetime import datetime, timedelta
from collections import deque
import requests
import re
import uuid
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
from database import SessionLocal
import models
import logging
import json
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- In-Memory Ring Buffer Log Handler ---

class MemoryLogHandler(logging.Handler):
    """Captures log entries in a ring buffer for the /api/logs endpoint.
    Uses deque (thread-safe for append/iteration) without extra locks
    to avoid deadlocks with uvicorn's internal logging."""
    MAX_ENTRIES = 500

    # Noisy loggers to exclude from the buffer
    _EXCLUDE_LOGGERS = {"uvicorn.access"}

    def __init__(self):
        super().__init__()
        self.entries = deque(maxlen=self.MAX_ENTRIES)

    def emit(self, record):
        if record.name in self._EXCLUDE_LOGGERS:
            return
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "source": record.name,
                "message": self.format(record),
            }
            self.entries.append(entry)
        except Exception:
            pass  # Never let log handler crash the app

    def get_entries(self, level=None, source=None, limit=200):
        items = list(self.entries)
        if level:
            items = [e for e in items if e["level"] == level.upper()]
        if source:
            items = [e for e in items if source.lower() in e["source"].lower() or source.lower() in e["message"].lower()]
        items.reverse()
        return items[:limit]

# Global log handler instance
log_handler = MemoryLogHandler()
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter("%(message)s"))

# Attach to root logger so ALL logs are captured
logging.getLogger().addHandler(log_handler)



EVENTS_DIR = "/events"
ENGINE_URL = os.getenv("ENGINE_URL", "http://engine:8000")
HA_URL = os.getenv("HA_URL")
HA_TOKEN = os.getenv("HA_TOKEN")
HA_SERVICE = os.getenv("HA_SERVICE", "/api/services/cover/open_cover")

# --- OCR Post-Processing ---

def normalize_plate_raw(plate_text: str) -> str:
    """Strip a plate down to its raw alphanumeric form for comparison.
    Removes dashes, spaces, and trailing E/H suffixes.
    Example: 'MK-MS-255E' -> 'MKMS255', 'WI-A-123-H' -> 'WIA123'
    """
    plate_text = plate_text.upper()
    plate_text = re.sub(r'[^A-ZÄÖÜ0-9]', '', plate_text)
    # Strip trailing E (Elektro) or H (Historisch) suffix for comparison
    plate_text = re.sub(r'[EH]$', '', plate_text)
    return plate_text

def apply_corrections(plate_text: str) -> str:
    """Clean OCR output and auto-format into German plate format with dashes.
    Handles trailing E (Elektro) and H (Historisch) suffixes.
    Examples:
      'MKMS255'   -> 'MK-MS-255'
      'MKMS255E'  -> 'MK-MS-255E'
      'WIA123H'   -> 'WI-A-123H'
      'B AB 1234' -> 'B-AB-1234'
    """
    plate_text = plate_text.upper()
    plate_text = re.sub(r'[^A-ZÄÖÜ0-9\-]', '', plate_text)

    # Auto-format German plates missing dashes
    if '-' not in plate_text:
        # Match with optional trailing E/H suffix (Elektro/Historisch)
        match = re.match(r'^([A-ZÄÖÜ]{1,3})([A-Z]{1,2})([0-9]{1,4})([EH]?)$', plate_text)
        if match:
            district = match.group(1)
            letters = match.group(2)
            numbers = match.group(3)
            suffix = match.group(4)  # 'E', 'H', or ''
            plate_text = f"{district}-{letters}-{numbers}{suffix}"

    # Fix plates that have dashes but E/H got separated by a dash (e.g. MK-MS-255-E -> MK-MS-255E)
    plate_text = re.sub(r'-([EH])$', r'\1', plate_text)

    return plate_text

def validate_plate(plate_text: str) -> bool:
    """Validate German plate format: XX-YY-1234 with optional E/H suffix."""
    pattern = r'^[A-ZÄÖÜ]{1,3}-[A-Z]{1,2}-[0-9]{1,4}[EH]?$'
    return bool(re.match(pattern, plate_text))

def parse_timestamp_from_filename(filename: str):
    """Extract timestamp from Blue Iris filename like einfahrttorCAR.20260223_150409412Z.jpg
    Returns datetime or None if can't parse."""
    match = re.search(r'(\d{8})_(\d{6})', filename)
    if match:
        try:
            date_str = match.group(1)  # 20260223
            time_str = match.group(2)  # 150409
            from datetime import datetime
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return None

SERIES_GAP_SECONDS = 9  # >9s gap between timestamps = new event

# --- Fuzzy Matching ---

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def fuzzy_match_plate(detected_plate: str, threshold: float = 0.80) -> tuple:
    """Match detected plate against DB plates with fuzzy matching.
    Comparison is done on RAW form (no dashes, no E/H suffix) so that
    'MKMS255' matches 'MK-MS-255E' and 'WIA123' matches 'WI-A-123H'.
    Returns: (matched_plate_text | None, match_score | None, decision)
    """
    if not detected_plate or detected_plate == "UNKNOWN":
        return None, None, "denied"

    detected_raw = normalize_plate_raw(detected_plate)
    if not detected_raw:
        return None, None, "denied"

    db = SessionLocal()
    try:
        plates = db.query(models.Plate).filter(models.Plate.active == True).all()
        best_match = None
        best_score = 0.0

        for plate in plates:
            plate_raw = normalize_plate_raw(plate.plate_text)
            max_len = max(len(detected_raw), len(plate_raw))
            if max_len == 0:
                continue
            distance = levenshtein_distance(detected_raw, plate_raw)
            score = 1.0 - (distance / max_len)

            if score > best_score:
                best_score = score
                best_match = plate.plate_text

        if best_match and best_score >= threshold:
            logger.info(f"Fuzzy match: '{detected_plate}' (raw: '{detected_raw}') -> '{best_match}' (score: {best_score:.2f})")
            return best_match, best_score, "allowed"
        else:
            logger.info(f"No fuzzy match for '{detected_plate}' (raw: '{detected_raw}') (best: {best_match}, score: {best_score:.2f})")
            return None, None, "denied"
    finally:
        db.close()

# --- Motion Burst State ---

SERIES_TIMEOUT = 30  # Wait up to 30s of silence before closing a series

HA_COOLDOWN = 60  # Only trigger Home Assistant once per minute per plate
MQTT_COOLDOWN = 60 # Only trigger MQTT once per minute per plate

active_series = {}  # {folder_path: {"series_id": sid, "event_id": eid, "last_activity": timestamp, "best_plate": plate, ...}}
series_lock = threading.Lock()
processed_files = set()
followup_timers = {}
timers_lock = threading.Lock()

last_ha_trigger = {}  # {plate_text: timestamp}
last_mqtt_trigger = {} # {plate_text: timestamp}


def safe_delete_file(file_path: str, max_retries: int = 5, delay: float = 0.5) -> bool:
    """Attempt to delete a file multiple times in case it's temporarily locked by another process (e.g. FTP)."""
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logger.warning(f"Could not delete {file_path} after {max_retries} attempts: {e}")
    return False

def process_single_image(img_path: str) -> dict:
    """Send a single image to the engine and return the best plate result.

    Returns dict with keys: plate, confidence, processing_time_ms, source, crop_jpeg, bbox
    crop_jpeg is extracted from the ORIGINAL full-resolution image using the engine's bbox.
    """
    # Wait for the file to finish writing (e.g. from FTP or Blue Iris)
    start_time = time.time()
    last_size = -1
    while time.time() - start_time < 2.0:
        try:
            current_size = os.path.getsize(img_path)
            if current_size > 0 and current_size == last_size:
                break
            last_size = current_size
        except OSError:
            pass
        time.sleep(0.1)

    try:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        files = {'file': (os.path.basename(img_path), img_bytes, 'image/jpeg')}
        response = requests.post(f"{ENGINE_URL}/analyze", files=files, timeout=10)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            processing_time_ms = data.get("processing_time_ms")

            best = {"plate": "", "confidence": 0.0, "processing_time_ms": processing_time_ms, "source": "fast_alpr", "crop_jpeg": None, "bbox": None}
            for res in results:
                plate_text = apply_corrections(res.get("plate", ""))
                conf = float(res.get("confidence", 0.0))
                if conf > best["confidence"]:
                    best = {
                        "plate": plate_text,
                        "confidence": conf,
                        "processing_time_ms": processing_time_ms,
                        "source": "fast_alpr",
                        "bbox": res.get("bbox"),
                        "crop_jpeg": None,
                    }
                    # Crop from original full-resolution image for best quality
                    bbox = res.get("bbox")
                    if bbox and len(bbox) == 4:
                        try:
                            import cv2
                            import numpy as np
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if orig_img is not None:
                                h, w = orig_img.shape[:2]
                                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                                # Clamp to image bounds
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                if x2 > x1 and y2 > y1:
                                    crop = orig_img[y1:y2, x1:x2]
                                    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                    best["crop_jpeg"] = buf.tobytes()
                        except Exception as e:
                            logger.warning(f"Could not crop plate from original image: {e}")
                    # Fallback: use engine's crop_b64 if own crop failed
                    if best["crop_jpeg"] is None:
                        import base64
                        crop_b64 = res.get("crop_b64")
                        if crop_b64:
                            best["crop_jpeg"] = base64.b64decode(crop_b64)
            return best
        else:
            logger.error(f"Engine API error {response.status_code} for {img_path}")
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
    return {"plate": "", "confidence": 0.0, "processing_time_ms": None, "source": "fast_alpr", "crop_jpeg": None, "bbox": None}

def get_setting(db, key: str, default: str) -> str:
    s = db.query(models.Setting).filter(models.Setting.key == key).first()
    return s.value if s else default

def trigger_ha(plate_text: str) -> bool:
    """Trigger Home Assistant for an allowed plate, with cooldown."""
    now = time.time()
    if plate_text in last_ha_trigger:
        elapsed = now - last_ha_trigger[plate_text]
        if elapsed < HA_COOLDOWN:
            logger.info(f"HA trigger skipped for '{plate_text}' — cooldown ({HA_COOLDOWN - elapsed:.0f}s remaining)")
            return False
    
    db = SessionLocal()
    try:
        ha_url = get_setting(db, "ha_url", HA_URL or "")
        ha_token = get_setting(db, "ha_token", HA_TOKEN or "")
        ha_service = get_setting(db, "ha_service", HA_SERVICE or "/api/services/cover/open_cover")
    finally:
        db.close()
    
    if not ha_url or not ha_token:
        logger.info("HA trigger skipped — no URL or Token configured")
        return False

    try:
        logger.info(f"Triggering HA for plate '{plate_text}'...")
        headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json"
        }
        ha_resp = requests.post(f"{ha_url}{ha_service}", headers=headers, json={}, timeout=5)
        logger.info(f"Home Assistant Response: {ha_resp.status_code}")
        last_ha_trigger[plate_text] = now
        return True
    except Exception as e:
        logger.error(f"Home Assistant trigger failed: {e}")
        return False

def trigger_mqtt(plate_text: str) -> bool:
    """Publish detected plate to MQTT broker, with cooldown."""
    now = time.time()
    if plate_text in last_mqtt_trigger:
        elapsed = now - last_mqtt_trigger[plate_text]
        if elapsed < MQTT_COOLDOWN:
            logger.info(f"MQTT trigger skipped for '{plate_text}' — cooldown ({MQTT_COOLDOWN - elapsed:.0f}s remaining)")
            return False
            
    db = SessionLocal()
    try:
        broker = get_setting(db, "mqtt_broker", "")
        port = int(get_setting(db, "mqtt_port", "1883"))
        user = get_setting(db, "mqtt_user", "")
        password = get_setting(db, "mqtt_pass", "")
        topic = get_setting(db, "mqtt_topic", "licenseplatecontrol/plate_detected")
    finally:
        db.close()
        
    if not broker:
        logger.info("MQTT trigger skipped — no broker configured")
        return False
        
    try:
        logger.info(f"Publishing MQTT for plate '{plate_text}'...")
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if user:
            client.username_pw_set(user, password)
        client.connect(broker, port, 5)
        from datetime import datetime # Import datetime here to avoid circular dependency if it's not already imported globally
        payload = json.dumps({"plate": plate_text, "timestamp": datetime.now().isoformat()})
        client.publish(topic, payload)
        client.disconnect()
        logger.info(f"MQTT published to topic {topic}")
        last_mqtt_trigger[plate_text] = now
        return True
    except Exception as e:
        logger.error(f"MQTT trigger failed: {e}")
        return False

def store_image_in_db(db, event_id: int, img_path: str, result: dict, is_trigger: bool = False):
    """Read image file, store in DB as EventImage, then delete the file."""
    try:
        with open(img_path, 'rb') as f:
            image_data = f.read()
        
        plate = result.get("plate", "")
        conf = result.get("confidence", 0.0)
        has_plate = bool(plate and plate != "UNKNOWN" and conf > 0.3)
        
        event_image = models.EventImage(
            event_id=event_id,
            filename=os.path.basename(img_path),
            image_data=image_data,
            detected_plate=plate if has_plate else None,
            confidence=conf if has_plate else None,
            has_plate=has_plate,
            is_trigger=is_trigger,
            processing_time_ms=result.get("processing_time_ms"),
            recognition_source=result.get("source", "fast_alpr"),
            plate_crop_data=result.get("crop_jpeg"),
        )
        db.add(event_image)
        db.flush()
        
        # Delete the source file safely
        if safe_delete_file(img_path):
            logger.info(f"Deleted processed image: {os.path.basename(img_path)}")
        
        return event_image
    except Exception as e:
        logger.error(f"Error storing image {img_path}: {e}")
        return None

def process_first_image(folder_path: str, img_path: str):
    """FAST-FIRST: Process the first image immediately, create event, trigger HA if needed."""
    logger.info(f"⚡ FAST-FIRST: Processing {os.path.basename(img_path)} immediately")

    cascade_settings = _get_cached_recognition_settings()
    rec_mode = cascade_settings["recognition_mode"]

    result = process_single_image(img_path)

    plate = result.get("plate", "")
    conf = result.get("confidence", 0.0)
    proc_time = result.get("processing_time_ms")

    proc_str = f" | {proc_time}ms" if proc_time else ""
    logger.info(f"[{os.path.basename(img_path)}] Detected: {plate} (Conf: {conf:.2f}){proc_str}")

    # Fuzzy match against DB
    matched_plate, match_score, decision = fuzzy_match_plate(plate)
    match_score_val = match_score if match_score is not None else 0.0

    # Hybrid cascade: PaddleOCR fallback when fuzzy-match is poor (not just OCR confidence)
    # Send only the plate CROP to PaddleOCR for better accuracy (not the full image)
    if rec_mode == "hybrid" and match_score_val < 0.80 and plate:
        logger.info(f"Hybrid Stage 2 (PaddleOCR) for first image — match_score {match_score_val:.2f} < 0.80")
        try:
            # Prefer plate crop from YOLO detection; fall back to full image
            crop_bytes = result.get("crop_jpeg")
            if crop_bytes:
                logger.info("PaddleOCR: Using YOLO plate crop (not full image)")
                paddle_input = crop_bytes
            else:
                logger.info("PaddleOCR: No crop available, using full image")
                with open(img_path, 'rb') as f:
                    paddle_input = f.read()
            paddle_result = analyze_with_paddleocr(paddle_input)
            p_plate = paddle_result.get("plate", "")
            if p_plate:
                p_matched, p_score, p_decision = fuzzy_match_plate(p_plate)
                p_score_val = p_score if p_score is not None else 0.0
                logger.info(f"PaddleOCR: '{p_plate}' (match={p_score_val:.2f}) vs fast_alpr: '{plate}' (match={match_score_val:.2f})")
                if p_score_val > match_score_val:
                    logger.info(f"PaddleOCR wins! '{plate}' -> '{p_plate}'")
                    # Keep the crop from fast_alpr (high quality)
                    crop_jpeg_backup = result.get("crop_jpeg")
                    result = paddle_result
                    result["crop_jpeg"] = crop_jpeg_backup
                    plate = result.get("plate", "")
                    conf = result.get("confidence", 0.0)
                    proc_time = result.get("processing_time_ms")
                    matched_plate, match_score, decision = p_matched, p_score, p_decision
        except Exception as e:
            logger.error(f"PaddleOCR fallback error: {e}")
    
    # Trigger integrations if allowed
    db = SessionLocal()
    ha_enabled = get_setting(db, "ha_enabled", "true").lower() == "true"
    mqtt_enabled = get_setting(db, "mqtt_enabled", "false").lower() == "true"
    db.close()
    
    ha_triggered = False
    mqtt_triggered = False
    trigger_ts = None
    if decision == "allowed":
        if ha_enabled:
            ha_triggered = trigger_ha(matched_plate or plate)
        if mqtt_enabled:
            mqtt_triggered = trigger_mqtt(matched_plate or plate)
            
        if ha_triggered or mqtt_triggered:
            trigger_ts = datetime.now()
    
    # Create event in DB
    sid = str(uuid.uuid4())[:8]
    
    db = SessionLocal()
    try:
        new_event = models.Event(
            detected_plate=plate,
            confidence=str(conf),
            decision=decision,
            series_id=sid,
            image_count=1,
            matched_plate=matched_plate,
            match_score=match_score,
            ha_triggered=ha_triggered,
            mqtt_triggered=mqtt_triggered,
            trigger_timestamp=trigger_ts,
            processing_time_ms=result.get("processing_time_ms"),
            recognition_source=rec_mode,
        )
        db.add(new_event)
        db.flush()  # Get the ID without committing
        event_id = new_event.id
        
        # Store image in DB and delete file — first image is the trigger if it detected a plate and decision is allowed
        has_plate = bool(plate and plate != 'UNKNOWN' and conf > 0.3)
        store_image_in_db(db, event_id, img_path, result, is_trigger=(has_plate and decision == 'allowed'))
        
        db.commit()
        logger.info(f"Event #{event_id} created: '{plate}' -> {decision.upper()} (series: {sid})")
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        event_id = None
    finally:
        db.close()
    
    # Register active series
    with series_lock:
        active_series[folder_path] = {
            "series_id": sid,
            "event_id": event_id,
            "last_activity": time.time(),
            "ha_triggered": ha_triggered,
            "best_plate": plate,
            "best_confidence": conf,
            "matched_plate": matched_plate,
            "match_score": match_score,
            "decision": decision,
            "mqtt_triggered": mqtt_triggered,
        }

def process_followup_images(folder_path: str):
    """Process follow-up images in a burst, update the existing event."""
    with timers_lock:
        if folder_path in followup_timers:
            del followup_timers[folder_path]
    
    all_images = glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                 glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]")) + \
                 glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]"))
    
    new_images = [img for img in all_images if img not in processed_files]
    if not new_images:
        return
    
    # We only process up to 10 images to save ML resources
    # Images beyond this limit must be deleted to prevent the folder from filling up
    image_files = sorted(new_images, key=os.path.getmtime, reverse=True)
    process_list = image_files[:10]
    process_list.reverse() # Restore chronological order (oldest to newest) for timeline display and processing
    ignore_list = image_files[10:]
    
    for img in new_images:
        processed_files.add(img)
        
    for img in ignore_list:
        if safe_delete_file(img):
            logger.info(f"Deleted overflow processed image: {os.path.basename(img)}")
    
    logger.info(f"📸 Follow-up: Processing {len(process_list)} images for series")
    
    with series_lock:
        series = active_series.get(folder_path)
    
    if not series or not series.get("event_id"):
        # No active series — treat as new first image
        if process_list:
            process_first_image(folder_path, process_list[0])
            for img in process_list[1:]:
                if safe_delete_file(img):
                    logger.info(f"Deleted unassigned image: {os.path.basename(img)}")
        return
    
    # Process each image and find best result
    best_plate = series["best_plate"]
    best_conf = series["best_confidence"]
    
    # Store per-image results for DB insertion
    cascade_settings = _get_cached_recognition_settings()
    rec_mode = cascade_settings["recognition_mode"]

    image_results = []
    for img_path in process_list:
        result = process_single_image(img_path)
        plate = result.get("plate", "")
        conf = result.get("confidence", 0.0)
        proc_time = result.get("processing_time_ms")
        proc_str = f" | {proc_time}ms" if proc_time else ""
        logger.info(f"  [{os.path.basename(img_path)}] Detected: {plate} (Conf: {conf:.2f}){proc_str}")
        image_results.append((img_path, result))

        if conf > best_conf and plate:
            best_plate = plate
            best_conf = conf

    # Fuzzy match the best fast_alpr result
    matched_plate_batch, match_score_batch, decision_batch = fuzzy_match_plate(best_plate)
    match_score_batch_val = match_score_batch if match_score_batch is not None else 0.0

    # Hybrid cascade: PaddleOCR fallback when fuzzy-match is poor
    # Send only the plate CROP to PaddleOCR for better accuracy
    if rec_mode == "hybrid" and match_score_batch_val < 0.80 and best_plate:
        logger.info(f"Hybrid Stage 2 (PaddleOCR) for follow-up batch — match_score {match_score_batch_val:.2f} < 0.80")
        try:
            # Find best crop from image results (prefer YOLO crop over full image)
            paddle_input = None
            for img_path, res in image_results:
                if res.get("crop_jpeg") and res.get("confidence", 0) > 0:
                    paddle_input = res["crop_jpeg"]
                    logger.info("PaddleOCR: Using YOLO plate crop from best detection")
                    break
            # Fallback: use full image file
            if paddle_input is None:
                best_img_path = image_results[0][0]
                for img_path, res in image_results:
                    if res.get("confidence", 0) > 0:
                        best_img_path = img_path
                        break
                if os.path.exists(best_img_path):
                    with open(best_img_path, 'rb') as f:
                        paddle_input = f.read()
                    logger.info("PaddleOCR: No crop available, using full image")
            if paddle_input:
                paddle_result = analyze_with_paddleocr(paddle_input)
                p_plate = paddle_result.get("plate", "")
                if p_plate:
                    p_matched, p_score, p_decision = fuzzy_match_plate(p_plate)
                    p_score_val = p_score if p_score is not None else 0.0
                    logger.info(f"PaddleOCR: '{p_plate}' (match={p_score_val:.2f}) vs fast_alpr: '{best_plate}' (match={match_score_batch_val:.2f})")
                    if p_score_val > match_score_batch_val:
                        best_plate = paddle_result["plate"]
                        best_conf = paddle_result["confidence"]
                        matched_plate_batch, match_score_batch, decision_batch = p_matched, p_score, p_decision
                        logger.info(f"PaddleOCR wins follow-up: '{best_plate}' (match={p_score_val:.2f})")
        except Exception as e:
            logger.error(f"PaddleOCR fallback error: {e}")

    # Re-check fuzzy match with the improved plate
    matched_plate, match_score, decision = fuzzy_match_plate(best_plate)
    
    db = SessionLocal()
    ha_enabled = get_setting(db, "ha_enabled", "true").lower() == "true"
    mqtt_enabled = get_setting(db, "mqtt_enabled", "false").lower() == "true"
    db.close()
    
    # If decision improved to allowed and triggers weren't fired yet, trigger now
    ha_triggered = series["ha_triggered"]
    mqtt_triggered = series.get("mqtt_triggered", False)
    if decision == "allowed":
        if ha_enabled and not ha_triggered:
            ha_triggered = trigger_ha(matched_plate or best_plate)
        if mqtt_enabled and not mqtt_triggered:
            mqtt_triggered = trigger_mqtt(matched_plate or best_plate)
    
    # Update event in DB and store images
    db = SessionLocal()
    try:
        event = db.query(models.Event).filter(models.Event.id == series["event_id"]).first()
        if event:
            event.detected_plate = best_plate
            event.confidence = str(best_conf)
            event.decision = decision
            event.matched_plate = matched_plate
            event.match_score = match_score
            event.ha_triggered = ha_triggered
            event.mqtt_triggered = mqtt_triggered
            
            if image_results and image_results[-1][1].get("processing_time_ms"):
                event.processing_time_ms = image_results[-1][1].get("processing_time_ms")
            
            # Store each follow-up image in DB
            for img_path, result in image_results:
                store_image_in_db(db, event.id, img_path, result)
            
            # Update image count from actual DB images
            img_count = db.query(models.EventImage).filter(models.EventImage.event_id == event.id).count()
            event.image_count = img_count
            
            db.commit()
            logger.info(f"Event #{event.id} updated: '{best_plate}' ({best_conf:.2f}) -> {decision.upper()} | {img_count} images")
    except Exception as e:
        db.rollback()
        logger.error(f"Database error updating event: {e}")
    finally:
        db.close()
    
    # Update series state
    with series_lock:
        if folder_path in active_series:
            active_series[folder_path].update({
                "last_activity": time.time(),
                "best_plate": best_plate,
                "best_confidence": best_conf,
                "ha_triggered": ha_triggered,
                "mqtt_triggered": mqtt_triggered,
                "matched_plate": matched_plate,
                "match_score": match_score,
                "decision": decision,
            })

def check_series_expiry():
    """Background thread: expire old series after SERIES_TIMEOUT seconds of silence."""
    while True:
        time.sleep(5)
        now = time.time()
        with series_lock:
            expired = [fp for fp, s in active_series.items() 
                      if now - s["last_activity"] > SERIES_TIMEOUT]
            for fp in expired:
                s = active_series.pop(fp)
                logger.info(f"Series {s['series_id']} expired (Event #{s['event_id']})")


def check_storage_cleanup():
    """Background thread: runs hourly limits cleanup based on settings."""
    while True:
        try:
            db = SessionLocal()
            days_known = int(get_setting(db, "cleanup_days_known", "30"))
            days_unknown = int(get_setting(db, "cleanup_days_unknown", "2"))
            
            now = datetime.now()
            cutoff_known = now - timedelta(days=days_known)
            cutoff_unknown = now - timedelta(days=days_unknown)
            
            # Find old UNKNOWN events
            old_unknown = db.query(models.Event).filter(
                models.Event.detected_plate == "UNKNOWN",
                models.Event.timestamp < cutoff_unknown
            ).all()
            
            # Find old KNOWN events
            old_known = db.query(models.Event).filter(
                models.Event.detected_plate != "UNKNOWN",
                models.Event.timestamp < cutoff_known
            ).all()
            
            events_to_delete = old_unknown + old_known
            
            if events_to_delete:
                logger.info(f"🧹 Storage Cleanup: Found {len(events_to_delete)} old events. Reclaiming space...")
                deleted_count = 0
                for event in events_to_delete:
                    # Delete child images (blob data) first to avoid FK errors & memory bloat
                    db.query(models.EventImage).filter(models.EventImage.event_id == event.id).delete(synchronize_session=False)
                    db.delete(event)
                    deleted_count += 1
                
                db.commit()
                logger.info(f"🧹 Storage Cleanup: Success. Deleted {deleted_count} events and their images.")
                
        except Exception as e:
            db.rollback()
            logger.error(f"Storage cleanup failed: {e}")
        finally:
            db.close()
            
        time.sleep(3600)  # Check once an hour


# --- RTSP Grabber with Smart Analysis ---

rtsp_status = {
    "state": "disabled",           # disabled | connecting | connected | error
    "message": "RTSP nicht aktiviert",
    "frames_grabbed": 0,           # Total frames read from stream
    "frames_analyzed": 0,          # Frames sent to engine
    "detections": 0,               # Frames where a real plate was found
    "frames_skipped": 0,           # Frames skipped (frame mode or buffer drain)
    "last_capture": None,          # ISO timestamp of last analysis
    "last_plate": None,            # Last detected plate text
    "last_confidence": None,       # Last detection confidence
    "last_processing_ms": None,    # Last engine processing time (ms)
    "last_analysis_timestamp": 0.0,# Unix time of last analysis (used for seconds mode buffer drain)
    "fps": 0.0,
    "url": "",
    "camera_name": "",
    "interval_mode": "seconds",
    "interval_value": 3,
    "backpressure": False,         # True if analysis takes longer than interval
    "recognition_mode": "fast_alpr",  # Current recognition mode
}

# Used to group identical vehicles into a single event instead of repeating DB writes
rtsp_active_session = {
    "is_active": False,
    "last_seen_time": 0.0,
    "images": [],
    "best_plate": "",
    "best_conf": 0.0,
    "best_match_score": 0.0,
    "decision": "DENIED",
    "has_triggered": False
}

# Latest RTSP preview frame (JPEG bytes) — served via API
rtsp_preview_frame = None
rtsp_unmasked_preview_frame = None

# Cache for the ROI mask to avoid recalculating np.zeros and cv2.fillPoly on every frame
rtsp_mask_cache = {
    "polygon_str": "",
    "frame_shape": None,
    "mask": None,
    "bbox": None, # (x, y, w, h)
    "cropped_mask": None, # mask cropped to bbox (for engine frame masking)
}

# Rolling average cache for processing times
rtsp_processing_times = deque(maxlen=10)

# --- Performance: Cached polygon setting (avoids DB query every frame) ---
_polygon_setting_cache = {"value": "", "last_check": 0.0}
_POLYGON_CACHE_TTL = 5.0  # seconds – polygon changes are rare, 5s is fine

def _get_cached_polygon() -> str:
    """Return the ROI polygon JSON string, refreshing from DB at most every 5 seconds."""
    now = time.time()
    if now - _polygon_setting_cache["last_check"] > _POLYGON_CACHE_TTL:
        try:
            db = SessionLocal()
            _polygon_setting_cache["value"] = get_setting(db, "rtsp_roi_polygon", "")
            db.close()
        except Exception:
            pass
        _polygon_setting_cache["last_check"] = now
    return _polygon_setting_cache["value"]

# Tracks last-seen timestamp per plate to prevent spam events (e.g., parked cars)
rtsp_cooldown = {}

# --- Debug Frame Buffer (1-hour rolling storage) ---
# Stores ALL analyzed frames with metadata for forensic debugging.
# Each entry: {"id": int, "timestamp": str, "jpeg": bytes, "plate": str, "confidence": float, "processing_ms": float}
debug_frame_buffer = deque(maxlen=3600)  # Max ~1h at 1fps
debug_frame_counter = 0
DEBUG_BUFFER_MAX_AGE_SECONDS = 3600  # 1 hour

def _store_debug_frame(jpeg_bytes, plate="", confidence=0.0, processing_ms=None):
    """Store a frame in the debug ring buffer with auto-purge of old entries."""
    global debug_frame_counter
    debug_frame_counter += 1
    entry = {
        "id": debug_frame_counter,
        "timestamp": datetime.now().isoformat(),
        "jpeg": jpeg_bytes,
        "plate": plate,
        "confidence": round(confidence, 3),
        "processing_ms": round(processing_ms, 1) if processing_ms else None,
    }
    debug_frame_buffer.append(entry)
    # Purge entries older than 1 hour
    cutoff = datetime.now() - timedelta(seconds=DEBUG_BUFFER_MAX_AGE_SECONDS)
    while debug_frame_buffer and datetime.fromisoformat(debug_frame_buffer[0]["timestamp"]) < cutoff:
        debug_frame_buffer.popleft()

def _analyze_frame_bytes(jpeg_bytes: bytes, filename: str = "frame.jpg") -> dict:
    """Send JPEG bytes to the engine API (single-pass /analyze) and return the best plate result.

    Used by the file watcher path. RTSP uses the 2-pass _detect_plates + _ocr_plate instead.
    Returns dict with keys: plate, confidence, processing_time_ms, source, crop_jpeg (optional bytes), bbox
    """
    try:
        files = {'file': (filename, jpeg_bytes, 'image/jpeg')}
        response = requests.post(f"{ENGINE_URL}/analyze", files=files, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            processing_time_ms = data.get("processing_time_ms")
            best = {"plate": "", "confidence": 0.0, "processing_time_ms": processing_time_ms, "source": "fast_alpr", "crop_jpeg": None, "bbox": None}
            for res in results:
                plate_text = apply_corrections(res.get("plate", ""))
                conf = float(res.get("confidence", 0.0))
                if conf > best["confidence"]:
                    best = {"plate": plate_text, "confidence": conf, "processing_time_ms": processing_time_ms, "source": "fast_alpr", "bbox": res.get("bbox"), "crop_jpeg": None}
                    # Fallback: use engine's crop if no bbox available for own cropping
                    crop_b64 = res.get("crop_b64")
                    if crop_b64:
                        import base64
                        best["crop_jpeg"] = base64.b64decode(crop_b64)
            return best
        else:
            logger.error(f"Engine API error {response.status_code} for RTSP frame")
    except Exception as e:
        logger.error(f"RTSP engine analysis error: {e}")
    return {"plate": "", "confidence": 0.0, "processing_time_ms": None, "source": "fast_alpr", "crop_jpeg": None, "bbox": None}


def _detect_plates(jpeg_bytes: bytes, filename: str = "frame.jpg") -> dict:
    """Pass 1: Send JPEG to engine /detect endpoint — YOLO detection only, no OCR.
    Returns dict with 'detections' list of {bbox, confidence} and 'processing_time_ms'."""
    try:
        files = {'file': (filename, jpeg_bytes, 'image/jpeg')}
        response = requests.post(f"{ENGINE_URL}/detect", files=files, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Engine /detect error {response.status_code}")
    except Exception as e:
        logger.error(f"Engine /detect request failed: {e}")
    return {"detections": [], "processing_time_ms": None}


def _ocr_plate(jpeg_bytes: bytes, filename: str = "crop.jpg") -> dict:
    """Pass 2: Send plate crop JPEG to engine /ocr endpoint — OCR only.
    Returns dict with 'plate', 'confidence', 'processing_time_ms'."""
    try:
        files = {'file': (filename, jpeg_bytes, 'image/jpeg')}
        response = requests.post(f"{ENGINE_URL}/ocr", files=files, timeout=10)
        if response.status_code == 200:
            data = response.json()
            plate = apply_corrections(data.get("plate", ""))
            data["plate"] = plate
            return data
        else:
            logger.error(f"Engine /ocr error {response.status_code}")
    except Exception as e:
        logger.error(f"Engine /ocr request failed: {e}")
    return {"plate": "", "confidence": 0.0, "processing_time_ms": None}


# --- PaddleOCR Backup Integration ---

_paddle_ocr_instance = None
_paddle_ocr_lock = threading.Lock()

def _get_paddle_ocr():
    """Lazy-initialize PaddleOCR (only loaded once, ~30 MB model download on first use)."""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        with _paddle_ocr_lock:
            if _paddle_ocr_instance is None:
                try:
                    from paddleocr import PaddleOCR
                    _paddle_ocr_instance = PaddleOCR(
                        use_angle_cls=True,
                        lang='en',  # Plates are alphanumeric
                        show_log=False,
                        use_gpu=False,
                    )
                    logger.info("PaddleOCR initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize PaddleOCR: {e}")
    return _paddle_ocr_instance


def analyze_with_paddleocr(jpeg_bytes: bytes) -> dict:
    """Run PaddleOCR on a single JPEG image as backup OCR.

    Returns:
        {"plate": str, "confidence": float, "processing_time_ms": float, "source": "paddleocr"}
    """
    import numpy as np
    import cv2

    ocr = _get_paddle_ocr()
    if ocr is None:
        return {"plate": "", "confidence": 0.0, "processing_time_ms": None, "source": "paddleocr"}

    start_time = time.time()
    try:
        # Decode JPEG to numpy array
        img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return {"plate": "", "confidence": 0.0, "processing_time_ms": None, "source": "paddleocr"}

        results = ocr.ocr(img, cls=True)
        processing_time_ms = (time.time() - start_time) * 1000.0

        if not results or not results[0]:
            logger.info(f"PaddleOCR: no text detected ({processing_time_ms:.0f}ms)")
            return {"plate": "", "confidence": 0.0, "processing_time_ms": processing_time_ms, "source": "paddleocr"}

        # Find the best plate-like text among all detected text regions
        best_plate = ""
        best_conf = 0.0
        for line in results[0]:
            text = line[1][0].strip().upper()
            conf = float(line[1][1])
            # Clean: keep only alphanumeric
            cleaned = re.sub(r'[^A-ZÄÖÜ0-9]', '', text)
            # Plate-like: 4-12 chars, has both letters and digits
            if 4 <= len(cleaned) <= 12 and re.search(r'[A-Z]', cleaned) and re.search(r'\d', cleaned):
                if conf > best_conf:
                    best_plate = cleaned
                    best_conf = conf

        if best_plate:
            best_plate = apply_corrections(best_plate)
            logger.info(f"PaddleOCR: plate='{best_plate}' conf={best_conf:.2f} ({processing_time_ms:.0f}ms)")
            return {
                "plate": best_plate,
                "confidence": best_conf,
                "processing_time_ms": processing_time_ms,
                "source": "paddleocr",
            }
        else:
            logger.info(f"PaddleOCR: no plate-like text found ({processing_time_ms:.0f}ms)")
            return {"plate": "", "confidence": 0.0, "processing_time_ms": processing_time_ms, "source": "paddleocr"}

    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000.0
        logger.error(f"PaddleOCR error: {e}")
        return {"plate": "", "confidence": 0.0, "processing_time_ms": processing_time_ms, "source": "paddleocr"}


# --- Recognition Settings (cached) ---

def _get_recognition_settings():
    """Read recognition settings from DB with defaults."""
    db = SessionLocal()
    try:
        return {
            "recognition_mode": get_setting(db, "recognition_mode", "fast_alpr"),
            "confidence_threshold": float(get_setting(db, "vision_llm_confidence_threshold", "0.6")),
        }
    finally:
        db.close()

_recognition_cache = {"settings": None, "last_check": 0.0}
_RECOGNITION_CACHE_TTL = 10.0

def _get_cached_recognition_settings():
    """Return cached recognition settings, refreshing from DB at most every 10 seconds."""
    now = time.time()
    if now - _recognition_cache["last_check"] > _RECOGNITION_CACHE_TTL or _recognition_cache["settings"] is None:
        _recognition_cache["settings"] = _get_recognition_settings()
        _recognition_cache["last_check"] = now
    return _recognition_cache["settings"]


rtsp_latest_frame_data = None
rtsp_frame_lock = threading.Lock()

def rtsp_reader_thread():
    """Producer Thread: Reads frames from RTSP as fast as possible to prevent FFMPEG buffering lag."""
    global rtsp_status, rtsp_latest_frame_data
    try:
        import cv2
    except ImportError as e:
        logger.error(f"📹 RTSP Reader: cv2 import failed — {e}")
        rtsp_status["state"] = "error"
        rtsp_status["message"] = f"OpenCV fehlt: {e}"
        return
    cap = None
    current_url = None
    consecutive_fails = 0
    MAX_FAILS_BEFORE_RECONNECT = 15

    while True:
        try:
            db = SessionLocal()
            enabled = get_setting(db, "rtsp_enabled", "false").lower() == "true"
            rtsp_url = get_setting(db, "rtsp_url", "")
            interval_value = int(get_setting(db, "rtsp_interval", "3"))
            interval_mode = get_setting(db, "rtsp_interval_mode", "seconds")
            camera_name = get_setting(db, "rtsp_camera_name", "cam1")
            db.close()

            rtsp_status["interval_mode"] = interval_mode
            rtsp_status["interval_value"] = interval_value
            rtsp_status["camera_name"] = camera_name

            if not enabled or not rtsp_url:
                if cap is not None:
                    cap.release()
                    cap = None
                    current_url = None
                    logger.info("📹 RTSP: Deaktiviert.")
                rtsp_status["state"] = "disabled" if not enabled else "error"
                rtsp_status["message"] = "RTSP nicht aktiviert" if not enabled else "Keine Stream-URL"
                rtsp_status["url"] = ""
                with rtsp_frame_lock:
                    rtsp_latest_frame_data = None
                time.sleep(5)
                continue

            rtsp_status["url"] = rtsp_url

            # (Re)connect
            if cap is None or current_url != rtsp_url:
                if cap is not None:
                    cap.release()
                rtsp_status["state"] = "connecting"
                rtsp_status["message"] = "Verbinde..."
                logger.info(f"📹 RTSP: Connecting to {rtsp_url}")

                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

                if cap.isOpened():
                    current_url = rtsp_url
                    consecutive_fails = 0
                    rtsp_status["state"] = "connected"
                    rtsp_status["message"] = "Verbunden — warte auf Frames..."
                    logger.info(f"📹 RTSP: Connected (TCP)!")
                else:
                    rtsp_status["state"] = "error"
                    rtsp_status["message"] = "Verbindung fehlgeschlagen (10s Retry)"
                    logger.warning(f"📹 RTSP: Connection failed to {rtsp_url}")
                    cap.release()
                    cap = None
                    time.sleep(10)
                    continue

            # Read frame as fast as possible
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_fails += 1
                if consecutive_fails >= MAX_FAILS_BEFORE_RECONNECT:
                    rtsp_status["state"] = "error"
                    rtsp_status["message"] = "Stream abgebrochen — Reconnecting..."
                    cap.release()
                    cap = None
                    current_url = None
                    consecutive_fails = 0
                    time.sleep(2)
                else:
                    time.sleep(0.1)
                continue

            consecutive_fails = 0
            rtsp_status["frames_grabbed"] += 1
            
            with rtsp_frame_lock:
                rtsp_latest_frame_data = frame

        except Exception as e:
            logger.error(f"📹 RTSP Reader Error: {e}")
            if cap is not None:
                cap.release()
                cap = None
                current_url = None
            time.sleep(5)

def rtsp_processor_thread():
    """Consumer Thread: Wakes up based on interval settings, grabs latest frame, analyzes it."""
    global rtsp_status, rtsp_preview_frame, rtsp_unmasked_preview_frame, rtsp_latest_frame_data, rtsp_active_session
    global rtsp_mask_cache, rtsp_processing_times
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        logger.error(f"📹 RTSP Processor: cv2/numpy import failed — {e}")
        return
    import json
    capture_times = []
    last_frame_grabbed = 0

    while True:
        try:
            if rtsp_status["state"] != "connected":
                time.sleep(1)
                continue

            mode = rtsp_status["interval_mode"]
            val = rtsp_status["interval_value"]
            cam_name = rtsp_status["camera_name"]

            # Wait for interval
            if mode == "frames":
                current_count = rtsp_status["frames_grabbed"]
                if current_count - last_frame_grabbed < max(val, 1):
                    time.sleep(0.05)
                    continue
                last_frame_grabbed = current_count
            else:
                # seconds
                if time.time() - rtsp_status["last_analysis_timestamp"] < val:
                    time.sleep(0.1)
                    continue
                rtsp_status["last_analysis_timestamp"] = time.time()

            # Grab latest frame thread-safely
            with rtsp_frame_lock:
                frame = rtsp_latest_frame_data
                if frame is not None:
                    frame = frame.copy() # Make a copy to avoid reading while writing
            
            if frame is None:
                time.sleep(0.1)
                continue

            # --- Analysis starts here ---
            _t0 = time.time()  # cycle start
            _t_mask = 0
            _t_enc = 0
            _t_preview = 0
            _t_engine = 0
            _t_match = 0
            _did_preview = False

            # ROI Masking (optimized: crop-first, then mask small region)
            polygon_str = _get_cached_polygon()  # cached, no DB query per frame

            if polygon_str:
                try:
                    points = json.loads(polygon_str)
                    if len(points) >= 3:
                        h, w = frame.shape[:2]

                        # Cache mechanism: only rebuild mask if polygon or frame shape changed
                        if (rtsp_mask_cache["polygon_str"] != polygon_str or
                            rtsp_mask_cache["frame_shape"] != frame.shape[:2]):

                            # Convert relative coordinates (0.0 to 1.0) to absolute pixels
                            poly_pts = np.array([
                                [int(p["x"] * w), int(p["y"] * h)]
                                for p in points
                            ], np.int32)
                            poly_pts = poly_pts.reshape((-1, 1, 2))

                            logger.info(f"Rebuilding ROI Mask Cache! Shape: {frame.shape}, Points: {points}")

                            # Create black mask and draw white polygon
                            mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(mask, [poly_pts], 255)

                            # Calculate Bounding Rect
                            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(poly_pts)
                            x_rect = max(0, x_rect)
                            y_rect = max(0, y_rect)
                            w_rect = min(w - x_rect, w_rect)
                            h_rect = min(h - y_rect, h_rect)

                            # Pre-compute cropped mask for engine (avoids re-slicing every frame)
                            cropped_mask = mask[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]

                            # Store in cache
                            rtsp_mask_cache["polygon_str"] = polygon_str
                            rtsp_mask_cache["frame_shape"] = frame.shape[:2]
                            rtsp_mask_cache["mask"] = mask
                            rtsp_mask_cache["bbox"] = (x_rect, y_rect, w_rect, h_rect)
                            rtsp_mask_cache["cropped_mask"] = cropped_mask

                        # OPTIMIZED: Crop first (cheap numpy view), then mask only the small cropped region
                        x_rect, y_rect, w_rect, h_rect = rtsp_mask_cache["bbox"]
                        frame_for_engine = frame[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect].copy()
                        cropped_mask = rtsp_mask_cache["cropped_mask"]
                        # cv2.bitwise_and is SIMD-optimized and much faster than numpy boolean indexing
                        frame_for_engine = cv2.bitwise_and(frame_for_engine, frame_for_engine, mask=cropped_mask)
                        has_roi = True
                    else:
                        frame_for_engine = frame
                        has_roi = False
                except Exception as e:
                    import traceback
                    logger.error(f"Failed to apply ROI mask: {e}")
                    logger.error(traceback.format_exc())
                    frame_for_engine = frame
                    has_roi = False
            else:
                frame_for_engine = frame
                has_roi = False

            _t_mask = time.time() - _t0

            # Resize ROI crop to max 640px width for YOLO (model expects 640px input anyway)
            # This avoids sending oversized crops while preserving detail from the full-res source
            eh, ew = frame_for_engine.shape[:2]
            YOLO_MAX_WIDTH = 640
            if ew > YOLO_MAX_WIDTH:
                scale = YOLO_MAX_WIDTH / ew
                new_w = YOLO_MAX_WIDTH
                new_h = int(eh * scale)
                frame_for_engine = cv2.resize(frame_for_engine, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.debug(f"Resized ROI crop {ew}x{eh} -> {new_w}x{new_h} for engine")

            # Single JPEG encode: only the engine frame (the part that matters)
            _t_enc_start = time.time()
            success_eng, eng_buf = cv2.imencode('.jpg', frame_for_engine, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success_eng:
                continue
            engine_bytes = eng_buf.tobytes()
            _t_enc = time.time() - _t_enc_start

            # Preview updates: rate-limited to max 2 FPS (saves 1-2 full-frame JPEG encodes per analysis)
            _now_preview = time.time()
            if _now_preview - rtsp_status.get("_last_preview_time", 0) > 0.5:
                _t_prev_start = time.time()
                _did_preview = True
                rtsp_status["_last_preview_time"] = _now_preview
                # Unmasked preview for ROI editor
                s_umb, b_umb = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if s_umb:
                    rtsp_unmasked_preview_frame = b_umb.tobytes()
                # Masked preview for live view
                if has_roi:
                    preview = cv2.bitwise_and(frame, frame, mask=rtsp_mask_cache["mask"])
                    s_prev, b_prev = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if s_prev:
                        rtsp_preview_frame = b_prev.tobytes()
                else:
                    rtsp_preview_frame = b_umb.tobytes() if s_umb else rtsp_preview_frame
                _t_preview = time.time() - _t_prev_start
            
            rtsp_status["frames_analyzed"] += 1
            rtsp_status["last_capture"] = datetime.now().isoformat()
            
            # FPS tracking
            now = time.time()
            capture_times.append(now)
            capture_times[:] = [t for t in capture_times if now - t < 60]
            if len(capture_times) > 1:
                elapsed = capture_times[-1] - capture_times[0]
                rtsp_status["fps"] = round(len(capture_times) / max(elapsed, 1), 2)

            # --- Recognition mode selection ---
            cascade_settings = _get_cached_recognition_settings()
            rec_mode = cascade_settings["recognition_mode"]
            rtsp_status["recognition_mode"] = rec_mode

            # === 2-PASS ARCHITECTURE ===
            # Pass 1: YOLO detection on 640px downscaled image (fast, ~100-200ms)
            # Pass 2: OCR on hi-res crop from original frame (~50-100ms)
            # Total: ~200-300ms instead of ~1500-3000ms single-pass
            analysis_start = time.time()

            detect_result = _detect_plates(engine_bytes, f"rtsp_{cam_name}.jpg")
            detections = detect_result.get("detections", [])
            detect_ms = detect_result.get("processing_time_ms")

            plate = ""
            confidence = 0.0
            ocr_ms = None
            result = {"plate": "", "confidence": 0.0, "processing_time_ms": None, "source": "fast_alpr", "crop_jpeg": None, "bbox": None}

            if detections:
                # Take best detection (highest confidence)
                best_det = max(detections, key=lambda d: d.get("confidence", 0))
                engine_bbox = best_det.get("bbox")

                if engine_bbox and len(engine_bbox) == 4:
                    ex1, ey1, ex2, ey2 = engine_bbox
                    # Reverse the resize: scale bbox back to ROI-crop size
                    if ew > YOLO_MAX_WIDTH:
                        inv_scale = 1.0 / scale  # scale was set during resize above
                    else:
                        inv_scale = 1.0
                    # Bbox in ROI-crop coordinate space
                    rx1 = int(ex1 * inv_scale)
                    ry1 = int(ey1 * inv_scale)
                    rx2 = int(ex2 * inv_scale)
                    ry2 = int(ey2 * inv_scale)
                    # Add ROI offset to get original frame coordinates
                    if has_roi:
                        ox1 = rx1 + x_rect
                        oy1 = ry1 + y_rect
                        ox2 = rx2 + x_rect
                        oy2 = ry2 + y_rect
                    else:
                        ox1, oy1, ox2, oy2 = rx1, ry1, rx2, ry2
                    # Clamp to original frame bounds
                    fh, fw = frame.shape[:2]
                    ox1, oy1 = max(0, ox1), max(0, oy1)
                    ox2, oy2 = min(fw, ox2), min(fh, oy2)

                    if ox2 > ox1 and oy2 > oy1:
                        # Hi-res crop from original 1080p frame
                        hires_crop = frame[oy1:oy2, ox1:ox2]
                        _, hires_buf = cv2.imencode('.jpg', hires_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        crop_jpeg_bytes = hires_buf.tobytes()
                        crop_h, crop_w = hires_crop.shape[:2]
                        logger.debug(f"2-pass: Hi-res crop {crop_w}x{crop_h}px from {fw}x{fh} frame")

                        # Pass 2: OCR on hi-res crop
                        ocr_result = _ocr_plate(crop_jpeg_bytes, f"rtsp_{cam_name}_crop.jpg")
                        plate = ocr_result.get("plate", "")
                        confidence = ocr_result.get("confidence", 0.0)
                        ocr_ms = ocr_result.get("processing_time_ms")

                        result = {
                            "plate": plate,
                            "confidence": confidence,
                            "processing_time_ms": (detect_ms or 0) + (ocr_ms or 0),
                            "source": "fast_alpr",
                            "crop_jpeg": crop_jpeg_bytes,
                            "bbox": [ox1, oy1, ox2, oy2],
                        }

            analysis_duration = time.time() - analysis_start
            _t_engine = analysis_duration
            proc_ms = result.get("processing_time_ms")

            if proc_ms:
                rtsp_processing_times.append(proc_ms)
                avg_ms = sum(rtsp_processing_times) / len(rtsp_processing_times)
                rtsp_status["last_processing_ms"] = round(avg_ms, 1)
            else:
                rtsp_status["last_processing_ms"] = None

            # Store in debug buffer (showing exactly what the engine saw)
            _store_debug_frame(engine_bytes, plate=plate, confidence=confidence, processing_ms=proc_ms)

            # Encode full original frame (not downscaled) for session storage
            _, orig_frame_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            orig_frame_bytes = orig_frame_buf.tobytes()

            # --- Intelligent Event Grouping Logic Starts Here ---
            is_real_plate = bool(plate and plate != "UNKNOWN" and confidence > 0.3)
            now_str = time.time()

            # Handle detection
            if is_real_plate:
                _t_match_start = time.time()
                matched_plate, match_score, decision = fuzzy_match_plate(plate)
                _t_match = time.time() - _t_match_start
                match_score_val = match_score if match_score is not None else 0.0
                
                # If no active session, start one
                if not rtsp_active_session["is_active"]:
                    rtsp_active_session["is_active"] = True
                    rtsp_active_session["last_seen_time"] = now_str
                    rtsp_active_session["images"] = []
                    rtsp_active_session["best_plate"] = plate
                    rtsp_active_session["best_conf"] = confidence
                    rtsp_active_session["best_match_score"] = match_score_val
                    rtsp_active_session["decision"] = decision
                    rtsp_active_session["has_triggered"] = False
                    rtsp_active_session["trigger_timestamp"] = None
                    logger.info(f"📹 RTSP: New vehicle session started with '{plate}' (conf={confidence:.2f}, mode={rec_mode})")
                else:
                    # Update active session timer
                    rtsp_active_session["last_seen_time"] = now_str
                    
                    # Update 'best' if this frame is better
                    if match_score_val > rtsp_active_session["best_match_score"] or (match_score_val == rtsp_active_session["best_match_score"] and confidence > rtsp_active_session["best_conf"]):
                        rtsp_active_session["best_plate"] = plate
                        rtsp_active_session["best_conf"] = confidence
                        rtsp_active_session["best_match_score"] = match_score_val
                        rtsp_active_session["decision"] = decision
                
                # --- FAST-FIRST TRIGGER LOGIC ---
                # Check if we should trigger immediately instead of waiting for 20s timeout
                if rtsp_active_session["decision"] == "ALLOWED" and not rtsp_active_session["has_triggered"]:
                    logger.info(f"📹 RTSP: Immediate trigger fired for '{rtsp_active_session['best_plate']}'")
                    ha_ok = trigger_ha(rtsp_active_session["best_plate"])
                    mq_ok = trigger_mqtt(rtsp_active_session["best_plate"])
                    rtsp_active_session["has_triggered"] = ha_ok or mq_ok
                    if rtsp_active_session["has_triggered"]:
                        rtsp_active_session["trigger_timestamp"] = datetime.now()
                        
                # Add current frame to session gallery (original resolution)
                rtsp_active_session["images"].append({
                    "jpeg_bytes": orig_frame_bytes,
                    "plate": plate,
                    "confidence": confidence,
                    "proc_ms": proc_ms,
                    "has_plate": True,
                    "source": result.get("source", "fast_alpr"),
                    "crop_jpeg": result.get("crop_jpeg"),
                })
                
                rtsp_status["state"] = "connected"
                mode_label = f"alle {val}s" if mode == "seconds" else f"jeder {val}. Frame"
                rtsp_status["message"] = f"Aktiv — Sammle Daten für: {rtsp_active_session['best_plate']} ({len(rtsp_active_session['images'])} Bilder) — {mode_label}"
                
            else:
                # No plate detected in this frame.
                
                # If we are in an active session, add empty frames up to a limit (so gallery shows the car leaving)
                if rtsp_active_session["is_active"] and len(rtsp_active_session["images"]) < 10:
                     rtsp_active_session["images"].append({
                        "jpeg_bytes": orig_frame_bytes,
                        "plate": "UNKNOWN",
                        "confidence": 0.0,
                        "proc_ms": proc_ms,
                        "has_plate": False,
                        "source": rec_mode,
                    })
                
                # Update status UI
                mode_label = f"alle {val}s" if mode == "seconds" else f"jeder {val}. Frame"
                rtsp_status["state"] = "connected"
                if rtsp_active_session["is_active"]:
                    time_left = 20 - int(now_str - rtsp_active_session["last_seen_time"])
                    rtsp_status["message"] = f"Aktiv — Auto verlässt Bild? (Abschluss in {time_left}s) — {mode_label}"
                else:
                    last_det = f" | Letzte: {rtsp_status['last_plate']}" if rtsp_status['last_plate'] else ""
                    rtsp_status["message"] = f"Aktiv — kein Kennzeichen{last_det} — {mode_label}"

            
            # --- Session Finalization (Timeout check) ---
            if rtsp_active_session["is_active"] and (now_str - rtsp_active_session["last_seen_time"] > 20):
                # 20 seconds passed without detecting a plate -> finalize event
                s_plate = rtsp_active_session["best_plate"]
                s_conf = rtsp_active_session["best_conf"]
                s_dec = rtsp_active_session["decision"]
                s_imgs = rtsp_active_session["images"]

                # --- HYBRID CASCADE: PaddleOCR fallback ---
                # Smart trigger: use fuzzy-match score, not OCR confidence.
                # If fast_alpr can't match a known plate (score < 0.80), PaddleOCR gets a chance.
                s_match_before = rtsp_active_session.get("best_match_score", 0.0)
                needs_paddle = rec_mode == "hybrid" and s_match_before < 0.80
                logger.info(f"📹 RTSP: Finalize — mode={rec_mode}, plate='{s_plate}', conf={s_conf:.2f}, match={s_match_before:.2f}, paddle={'YES' if needs_paddle else 'skip'}")

                if needs_paddle:
                    plate_frames = [img for img in s_imgs if img.get("has_plate")]
                    # Sort by confidence desc — try best frame first
                    plate_frames.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                    # Prefer plate crop from YOLO detection for PaddleOCR
                    best_crop = None
                    for pf in plate_frames:
                        if pf.get("crop_jpeg"):
                            best_crop = pf["crop_jpeg"]
                            break
                    if best_crop:
                        paddle_input = best_crop
                        logger.info(f"📹 RTSP: PaddleOCR using YOLO plate crop (not full frame)")
                    else:
                        paddle_input = plate_frames[0]["jpeg_bytes"] if plate_frames else (s_imgs[0]["jpeg_bytes"] if s_imgs else None)
                        logger.info(f"📹 RTSP: PaddleOCR using full frame (no crop available)")

                    if paddle_input:
                        logger.info(f"📹 RTSP: Hybrid Stage 2 (PaddleOCR) — fast_alpr match only {s_match_before:.2f}")
                        paddle_result = analyze_with_paddleocr(paddle_input)
                        p_plate = paddle_result.get("plate", "")
                        p_conf = paddle_result.get("confidence", 0.0)

                        if p_plate:
                            # Check if PaddleOCR result fuzzy-matches better
                            p_matched, p_score, p_decision = fuzzy_match_plate(p_plate)
                            p_score_val = p_score if p_score is not None else 0.0
                            logger.info(f"📹 RTSP: PaddleOCR result: '{p_plate}' (conf={p_conf:.2f}, match={p_score_val:.2f}) vs fast_alpr: '{s_plate}' (match={s_match_before:.2f})")

                            if p_score_val > s_match_before:
                                logger.info(f"📹 RTSP: PaddleOCR wins! '{s_plate}' -> '{p_plate}' (match {s_match_before:.2f} -> {p_score_val:.2f})")
                                s_plate = apply_corrections(p_plate)
                                s_conf = p_conf

                # Re-do match just to be safe it's correct for the final selected plate
                s_matched_plate, s_match_score, s_dec_final = fuzzy_match_plate(s_plate)

                logger.info(f"📹 RTSP: Finalizing session for '{s_plate}' after 20s timeout. {len(s_imgs)} frames collected.")
                
                rtsp_status["detections"] += 1
                rtsp_status["last_plate"] = s_plate
                rtsp_status["last_confidence"] = round(s_conf, 2)
                
                db = SessionLocal()
                try:
                    event = models.Event(
                        timestamp=datetime.now(),
                        detected_plate=s_plate,
                        confidence=f"{s_conf:.4f}",
                        decision=s_dec_final,
                        series_id=f"rtsp_{cam_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        image_count=len(s_imgs),
                        matched_plate=s_matched_plate,
                        match_score=s_match_score,
                        ha_triggered=rtsp_active_session.get("has_triggered", False),
                        trigger_timestamp=rtsp_active_session.get("trigger_timestamp"),
                        processing_time_ms=s_imgs[0]["proc_ms"] if s_imgs else None,
                        recognition_source=rec_mode,
                    )
                    db.add(event)
                    db.flush()

                    for i, img_data in enumerate(s_imgs):
                        event_image = models.EventImage(
                            event_id=event.id,
                            filename=f"rtsp_{cam_name}_{datetime.now().strftime('%H%M%S')}_{i}.jpg",
                            image_data=img_data["jpeg_bytes"],
                            detected_plate=img_data["plate"],
                            confidence=img_data["confidence"],
                            has_plate=img_data["has_plate"],
                            is_trigger=(i == 0),
                            processing_time_ms=img_data["proc_ms"],
                            recognition_source=img_data.get("source", "fast_alpr"),
                            plate_crop_data=img_data.get("crop_jpeg"),
                        )
                        db.add(event_image)
                    
                    db.commit()
                    logger.info(f"📹 RTSP: Consolidated Event #{event.id} created successfully!")

                    # Note: HA/MQTT trigger was already fired immediately upon first match!

                except Exception as e:
                    db.rollback()
                    logger.error(f"📹 RTSP: DB error during session finalization: {e}")
                finally:
                    db.close()
                
                # Reset session
                rtsp_active_session = {
                    "is_active": False,
                    "last_seen_time": 0.0,
                    "images": [],
                    "best_plate": "",
                    "best_conf": 0.0,
                    "best_match_score": 0.0,
                    "decision": "DENIED",
                    "has_triggered": False
                }

            _t_total = time.time() - _t0
            if mode == "seconds" and _t_total > val:
                rtsp_status["backpressure"] = True
            else:
                rtsp_status["backpressure"] = False

            # Performance logging: DEBUG for every frame, WARNING only for sustained backpressure
            _eng_ms = int(proc_ms) if proc_ms else int(_t_engine * 1000)
            _net_ms = int((_t_engine - proc_ms / 1000) * 1000) if proc_ms else 0
            _total_ms = int(_t_total * 1000)

            # Track consecutive slow frames for real backpressure detection
            _interval_ms = int(val * 1000) if mode == "seconds" and val > 0 else 5000
            if _total_ms > _interval_ms:
                rtsp_status["_slow_frames"] = rtsp_status.get("_slow_frames", 0) + 1
            else:
                rtsp_status["_slow_frames"] = 0

            # Only warn if 3+ consecutive frames exceed the interval (real backpressure)
            if rtsp_status.get("_slow_frames", 0) >= 3:
                parts = [
                    f"BACKPRESSURE {_total_ms}ms (limit {_interval_ms}ms, {rtsp_status['_slow_frames']}x slow)",
                    f"engine={_eng_ms}(net={_net_ms})",
                    f"plate={plate or '-'}",
                ]
                logger.warning(f"📹 {' | '.join(parts)}")
            else:
                # Normal performance trace at DEBUG level (won't flood logs)
                logger.debug(f"📹 cycle {_total_ms}ms | engine={_eng_ms} | plate={plate or '-'}")

        except Exception as e:
            logger.error(f"📹 RTSP Processor Error: {e}")
            time.sleep(5)


class EventFolderHandler(FileSystemEventHandler):
    def process_path(self, path):
        ext = path.lower().split('.')[-1]
        if ext not in ['jpg', 'jpeg', 'png']:
            return
        
        if path in processed_files:
            return
        
        folder_path = os.path.dirname(path)
        
        try:
            with series_lock:
                series = active_series.get(folder_path)
                is_new_series = (series is None or 
                                 time.time() - series["last_activity"] > SERIES_TIMEOUT)
            
            if is_new_series:
                # FAST-FIRST: Process immediately!
                processed_files.add(path)
                process_first_image(folder_path, path)
            else:
                # Follow-up image: debounce 2s then batch process
                with timers_lock:
                    if folder_path in followup_timers:
                        followup_timers[folder_path].cancel()
                    timer = threading.Timer(2.0, process_followup_images, args=[folder_path])
                    followup_timers[folder_path] = timer
                    timer.start()
                
                # Update series timestamp
                with series_lock:
                    if folder_path in active_series:
                        active_series[folder_path]["last_activity"] = time.time()
        except Exception as e:
            logger.error(f"Error evaluating path {path}: {e}")

    def on_created(self, event):
        self.process_path(event.src_path)
        
    def on_modified(self, event):
        self.process_path(event.src_path)

def group_images_by_timestamp(img_paths: list) -> list:
    """Group images into series based on filename timestamps.
    Returns list of lists, where each inner list is one series.
    Uses >9s gap to split into separate events."""
    # Sort by parsed timestamp (fallback to mtime)
    def sort_key(path):
        ts = parse_timestamp_from_filename(os.path.basename(path))
        if ts:
            return ts.timestamp()
        return os.path.getmtime(path)
    
    sorted_imgs = sorted(img_paths, key=sort_key)
    if not sorted_imgs:
        return []
    
    series_list = [[sorted_imgs[0]]]
    prev_ts = sort_key(sorted_imgs[0])
    
    for img_path in sorted_imgs[1:]:
        curr_ts = sort_key(img_path)
        gap = curr_ts - prev_ts
        
        if gap > SERIES_GAP_SECONDS:
            # Start a new series
            series_list.append([img_path])
        else:
            series_list[-1].append(img_path)
        prev_ts = curr_ts
    
    return series_list

def process_startup_series(folder_path: str, imgs: list):
    """Process a single series of images found at startup."""
    logger.info(f"📥 Importing series: {len(imgs)} images")
    
    # Mark all as processed so file watcher doesn't re-trigger
    for img in imgs:
        processed_files.add(img)
    
    # Analyze all images
    image_results = []
    best_plate = ""
    best_conf = 0.0
    first_trigger_idx = None  # Index of first image that detected a plate
    
    for i, img_path in enumerate(imgs):
        result = process_single_image(img_path)
        plate = result.get("plate", "")
        conf = result.get("confidence", 0.0)
        has_plate = bool(plate and plate != "UNKNOWN" and conf > 0.3)
        proc_time = result.get("processing_time_ms")
        proc_str = f" | {proc_time}ms" if proc_time else ""
        logger.info(f"  [{os.path.basename(img_path)}] Detected: {plate} (Conf: {conf:.2f}){proc_str}")
        image_results.append((img_path, result))
        
        if has_plate and first_trigger_idx is None:
            first_trigger_idx = i
        
        if conf > best_conf and plate:
            best_plate = plate
            best_conf = conf
    
    # Determine decision
    matched_plate, match_score, decision = fuzzy_match_plate(best_plate)
    
    db = SessionLocal()
    ha_enabled = get_setting(db, "ha_enabled", "true").lower() == "true"
    mqtt_enabled = get_setting(db, "mqtt_enabled", "false").lower() == "true"
    db.close()
    
    ha_triggered = False
    mqtt_triggered = False
    if decision == "allowed":
        if ha_enabled:
            ha_triggered = trigger_ha(matched_plate or best_plate)
        if mqtt_enabled:
            mqtt_triggered = trigger_mqtt(matched_plate or best_plate)
    
    # Create event and store all images
    sid = str(uuid.uuid4())[:8]
    db = SessionLocal()
    try:
        last_result = image_results[-1][1] if image_results else {}
        new_event = models.Event(
            detected_plate=best_plate or "UNKNOWN",
            confidence=str(best_conf),
            decision=decision,
            series_id=sid,
            image_count=len(imgs),
            matched_plate=matched_plate,
            match_score=match_score,
            ha_triggered=ha_triggered,
            mqtt_triggered=mqtt_triggered,
            processing_time_ms=last_result.get("processing_time_ms")
        )
        db.add(new_event)
        db.flush()
        event_id = new_event.id
        
        for i, (img_path, result) in enumerate(image_results):
            is_trigger = (i == first_trigger_idx and decision == 'allowed')
            store_image_in_db(db, event_id, img_path, result, is_trigger=is_trigger)
        
        db.commit()
        logger.info(f"Event #{event_id} created: '{best_plate}' ({best_conf:.2f}) -> {decision.upper()} | {len(imgs)} images (series: {sid})")
    except Exception as e:
        db.rollback()
        logger.error(f"Database error on startup import: {e}")
    finally:
        db.close()

def _startup_system_check():
    """Run system checks at startup and log detailed status."""
    logger.info("=" * 60)
    logger.info("  LicensePlateControl Backend v1.5.0 — System Start")
    logger.info("=" * 60)

    # 1. Engine health check
    try:
        resp = requests.get(f"{ENGINE_URL}/health", timeout=5)
        if resp.status_code == 200:
            health = resp.json()
            alpr_ok = health.get("alpr_loaded", False)
            device = health.get("device", "?")
            version = health.get("version", "?")
            mock = health.get("mock_mode", False)
            if alpr_ok:
                logger.info(f"  Engine: OK — v{version} | ALPR loaded | Device: {device}")
            else:
                err = health.get("error", "unknown")
                logger.warning(f"  Engine: DEGRADED — ALPR not loaded ({err}) | MOCK mode active")
        else:
            logger.warning(f"  Engine: HTTP {resp.status_code} — may not be ready yet")
    except Exception as e:
        logger.warning(f"  Engine: UNREACHABLE — {e} (will retry on first image)")

    # 2. Recognition mode
    db = SessionLocal()
    try:
        rec_mode = get_setting(db, "recognition_mode", "fast_alpr")
        ha_enabled = get_setting(db, "ha_enabled", "true").lower() == "true"
        mqtt_enabled = get_setting(db, "mqtt_enabled", "false").lower() == "true"
        rtsp_enabled = get_setting(db, "rtsp_enabled", "false").lower() == "true"
        rtsp_url = get_setting(db, "rtsp_url", "")
        folder_watch = get_setting(db, "folder_watch_enabled", "true").lower() == "true"
        plate_count = db.query(models.Plate).filter(models.Plate.active == True).count()
        event_count = db.query(models.Event).count()
    finally:
        db.close()

    logger.info(f"  Recognition Mode: {rec_mode.upper()}")
    if rec_mode == "hybrid":
        logger.info(f"  Hybrid: fast_alpr (primary) + PaddleOCR (fallback on match < 0.80)")
        logger.info(f"  PaddleOCR: Will initialize on first use (lazy-load)")
    logger.info(f"  Active Plates: {plate_count} | Stored Events: {event_count}")
    logger.info(f"  Integrations: HA={'ON' if ha_enabled else 'OFF'} | MQTT={'ON' if mqtt_enabled else 'OFF'}")
    logger.info(f"  Inputs: Folder Watch={'ON' if folder_watch else 'OFF'} | RTSP={'ON' if rtsp_enabled else 'OFF'}")
    if rtsp_enabled and rtsp_url:
        logger.info(f"  RTSP URL: {rtsp_url}")
    logger.info("=" * 60)


def start_watcher():
    if not os.path.exists(EVENTS_DIR):
        os.makedirs(EVENTS_DIR, exist_ok=True)

    # Run startup system check
    _startup_system_check()

    # Find all existing images on disk
    existing = glob.glob(os.path.join(EVENTS_DIR, "**", "*.[jJ][pP][gG]"), recursive=True) + \
               glob.glob(os.path.join(EVENTS_DIR, "**", "*.[jJ][pP][eE][gG]"), recursive=True) + \
               glob.glob(os.path.join(EVENTS_DIR, "**", "*.[pP][nN][gG]"), recursive=True)
    
    # Check DB for already-imported filenames
    db = SessionLocal()
    try:
        imported_filenames = set()
        all_imgs = db.query(models.EventImage.filename).all()
        for row in all_imgs:
            imported_filenames.add(row[0])
    finally:
        db.close()
    
    # Separate into already-imported (skip) vs unimported (process)
    unimported = []
    for img in existing:
        basename = os.path.basename(img)
        if basename in imported_filenames:
            processed_files.add(img)
        else:
            unimported.append(img)
    
    logger.info(f"Found {len(existing)} images on disk, {len(imported_filenames)} already in DB, {len(unimported)} new to process.")
    
    # Process unimported images — group by folder, then split by timestamp gaps
    if unimported:
        from collections import defaultdict
        by_folder = defaultdict(list)
        for img_path in unimported:
            folder = os.path.dirname(img_path)
            by_folder[folder].append(img_path)
        
        for folder_path, imgs in by_folder.items():
            series_groups = group_images_by_timestamp(imgs)
            logger.info(f"📥 Startup: {len(imgs)} images in {os.path.basename(folder_path) or 'root'} -> {len(series_groups)} event(s)")
            
            for series_imgs in series_groups:
                process_startup_series(folder_path, series_imgs)
    
    # Start series expiry and cleanup background threads
    expiry_thread = threading.Thread(target=check_series_expiry, daemon=True)
    expiry_thread.start()
    
    cleanup_thread = threading.Thread(target=check_storage_cleanup, daemon=True)
    cleanup_thread.start()
    
    # Start RTSP Reader & Processor threads
    reader_thread = threading.Thread(target=rtsp_reader_thread, daemon=True)
    reader_thread.start()
    
    processor_thread = threading.Thread(target=rtsp_processor_thread, daemon=True)
    processor_thread.start()
    
    # Start Folder Watcher Manager thread
    manager_thread = threading.Thread(target=folder_watch_manager, daemon=True)
    manager_thread.start()
    
    logger.info("⚡ Background services initialized.")
    return None

def folder_watch_manager():
    """Background thread to start/stop the folder observer based on database settings."""
    observer = None
    event_handler = EventFolderHandler()
    
    while True:
        try:
            db = SessionLocal()
            enabled = get_setting(db, "folder_watch_enabled", "true").lower() == "true"
            db.close()
            
            if enabled and observer is None:
                observer = Observer()
                observer.schedule(event_handler, EVENTS_DIR, recursive=True)
                observer.start()
                logger.info(f"📂 Folder Watcher: Started monitoring {EVENTS_DIR}")
            
            elif not enabled and observer is not None:
                observer.stop()
                observer.join()
                observer = None
                logger.info("📂 Folder Watcher: Stopped monitoring.")
                
        except Exception as e:
            logger.error(f"📂 Folder Watcher Manager Error: {e}")
            
        time.sleep(10)

