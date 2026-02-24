import time
import os
import glob
import threading
from datetime import datetime, timedelta
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

EVENTS_DIR = "/events"
ENGINE_URL = os.getenv("ENGINE_URL", "http://engine:8000")
HA_URL = os.getenv("HA_URL")
HA_TOKEN = os.getenv("HA_TOKEN")
HA_SERVICE = os.getenv("HA_SERVICE", "/api/services/cover/open_cover")

# --- OCR Post-Processing ---

def apply_corrections(plate_text: str) -> str:
    plate_text = plate_text.upper()
    plate_text = re.sub(r'[^A-ZÄÖÜ0-9\-]', '', plate_text)
    # Auto-format German plates missing dashes (e.g. MKMS255 -> MK-MS-255)
    if '-' not in plate_text:
        match = re.match(r'^([A-ZÄÖÜ]{1,3})([A-Z]{1,2})([0-9]{1,4})$', plate_text)
        if match:
            plate_text = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return plate_text

def validate_plate(plate_text: str) -> bool:
    pattern = r'^[A-ZÄÖÜ]{1,3}-[A-Z]{1,2}-[0-9]{1,4}$'
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
    Returns: (matched_plate_text | None, match_score | None, decision)
    """
    if not detected_plate or detected_plate == "UNKNOWN":
        return None, None, "denied"
    
    db = SessionLocal()
    try:
        plates = db.query(models.Plate).filter(models.Plate.active == True).all()
        best_match = None
        best_score = 0.0
        
        for plate in plates:
            max_len = max(len(detected_plate), len(plate.plate_text))
            if max_len == 0:
                continue
            distance = levenshtein_distance(detected_plate, plate.plate_text)
            score = 1.0 - (distance / max_len)
            
            if score > best_score:
                best_score = score
                best_match = plate.plate_text
        
        if best_match and best_score >= threshold:
            logger.info(f"Fuzzy match: '{detected_plate}' -> '{best_match}' (score: {best_score:.2f})")
            return best_match, best_score, "allowed"
        else:
            logger.info(f"No fuzzy match for '{detected_plate}' (best: {best_match}, score: {best_score:.2f})")
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


def process_single_image(img_path: str) -> dict:
    """Send a single image to the engine and return the best plate result."""
    try:
        with open(img_path, 'rb') as f:
            files = {'file': (os.path.basename(img_path), f, 'image/jpeg')}
            response = requests.post(f"{ENGINE_URL}/analyze", files=files, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            processing_time_ms = data.get("processing_time_ms")
            
            best = {"plate": "", "confidence": 0.0, "processing_time_ms": processing_time_ms}
            for res in results:
                plate_text = apply_corrections(res.get("plate", ""))
                conf = float(res.get("confidence", 0.0))
                if conf > best["confidence"]:
                    best = {
                        "plate": plate_text, 
                        "confidence": conf,
                        "processing_time_ms": processing_time_ms
                    }
            return best
        else:
            logger.error(f"Engine API error {response.status_code} for {img_path}")
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
    return {"plate": "", "confidence": 0.0, "processing_time_ms": None}

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
        payload = json.dumps({"plate": plate_text, "timestamp": datetime.utcnow().isoformat()})
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
            is_trigger=is_trigger
        )
        db.add(event_image)
        db.flush()
        
        # Delete the source file
        try:
            os.remove(img_path)
            logger.info(f"Deleted processed image: {os.path.basename(img_path)}")
        except OSError as e:
            logger.warning(f"Could not delete {img_path}: {e}")
        
        return event_image
    except Exception as e:
        logger.error(f"Error storing image {img_path}: {e}")
        return None

def process_first_image(folder_path: str, img_path: str):
    """FAST-FIRST: Process the first image immediately, create event, trigger HA if needed."""
    logger.info(f"⚡ FAST-FIRST: Processing {os.path.basename(img_path)} immediately")
    
    result = process_single_image(img_path)
    plate = result["plate"]
    conf = result["confidence"]
    
    logger.info(f"[{os.path.basename(img_path)}] Detected: {plate} (Conf: {conf:.2f})")
    
    # Fuzzy match against DB
    matched_plate, match_score, decision = fuzzy_match_plate(plate)
    
    # Trigger integrations if allowed
    db = SessionLocal()
    ha_enabled = get_setting(db, "ha_enabled", "true").lower() == "true"
    mqtt_enabled = get_setting(db, "mqtt_enabled", "false").lower() == "true"
    db.close()
    
    ha_triggered = False
    mqtt_triggered = False
    if decision == "allowed":
        if ha_enabled:
            ha_triggered = trigger_ha(matched_plate or plate)
        if mqtt_enabled:
            mqtt_triggered = trigger_mqtt(matched_plate or plate)
    
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
            processing_time_ms=result.get("processing_time_ms")
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
    
    for img in new_images:
        processed_files.add(img)
    
    image_files = sorted(new_images, key=os.path.getmtime, reverse=True)[:10]
    logger.info(f"📸 Follow-up: Processing {len(image_files)} images for series")
    
    with series_lock:
        series = active_series.get(folder_path)
    
    if not series or not series.get("event_id"):
        # No active series — treat as new first image
        if image_files:
            process_first_image(folder_path, image_files[0])
            for img in image_files[1:]:
                processed_files.add(img)
        return
    
    # Process each image and find best result
    best_plate = series["best_plate"]
    best_conf = series["best_confidence"]
    
    # Store per-image results for DB insertion
    image_results = []
    for img_path in image_files:
        result = process_single_image(img_path)
        plate = result["plate"]
        conf = result["confidence"]
        logger.info(f"  [{os.path.basename(img_path)}] Detected: {plate} (Conf: {conf:.2f})")
        image_results.append((img_path, result))
        
        if conf > best_conf and plate:
            best_plate = plate
            best_conf = conf
    
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
            
            now = datetime.utcnow()
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


# --- RTSP Grabber with Status Tracking ---

rtsp_status = {
    "state": "disabled",       # disabled | connecting | connected | error | stopped
    "message": "RTSP nicht aktiviert",
    "frames_captured": 0,
    "frames_analyzed": 0,
    "last_capture": None,
    "fps": 0.0,
    "url": "",
    "camera_name": "",
    "interval_mode": "seconds",
    "interval_value": 3,
}

def rtsp_grabber():
    """Background thread: grab frames from an RTSP stream at configurable intervals."""
    global rtsp_status
    import cv2
    
    cap = None
    current_url = None
    consecutive_fails = 0
    MAX_FAILS_BEFORE_RECONNECT = 5
    frame_counter = 0  # Total frames read from stream (for frame-skip mode)
    capture_times = []  # For FPS calculation
    
    while True:
        try:
            db = SessionLocal()
            enabled = get_setting(db, "rtsp_enabled", "false").lower() == "true"
            rtsp_url = get_setting(db, "rtsp_url", "")
            interval_value = int(get_setting(db, "rtsp_interval", "3"))
            interval_mode = get_setting(db, "rtsp_interval_mode", "seconds")  # "seconds" or "frames"
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
                    logger.info("📹 RTSP Grabber: Disabled or no URL. Sleeping...")
                rtsp_status["state"] = "disabled" if not enabled else "error"
                rtsp_status["message"] = "RTSP nicht aktiviert" if not enabled else "Keine Stream-URL konfiguriert"
                rtsp_status["url"] = ""
                time.sleep(10)
                continue
            
            rtsp_status["url"] = rtsp_url
            
            # (Re)connect if URL changed or no connection
            if cap is None or current_url != rtsp_url:
                if cap is not None:
                    cap.release()
                rtsp_status["state"] = "connecting"
                rtsp_status["message"] = f"Verbinde mit {rtsp_url} ..."
                logger.info(f"📹 RTSP Grabber: Connecting to {rtsp_url} ...")
                
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if cap.isOpened():
                    current_url = rtsp_url
                    consecutive_fails = 0
                    frame_counter = 0
                    mode_label = f"alle {interval_value}s" if interval_mode == "seconds" else f"jeder {interval_value}. Frame"
                    rtsp_status["state"] = "connected"
                    rtsp_status["message"] = f"Verbunden — Modus: {mode_label}"
                    logger.info(f"📹 RTSP Grabber: Connected! Mode: {interval_mode}, Value: {interval_value}, Camera: {camera_name}")
                else:
                    rtsp_status["state"] = "error"
                    rtsp_status["message"] = f"Verbindung fehlgeschlagen. Neuer Versuch in 15s..."
                    logger.warning(f"📹 RTSP Grabber: Failed to connect to {rtsp_url}. Retrying in 15s...")
                    cap.release()
                    cap = None
                    time.sleep(15)
                    continue
            
            # Grab a frame
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_fails += 1
                rtsp_status["message"] = f"Frame-Fehler ({consecutive_fails}/{MAX_FAILS_BEFORE_RECONNECT})"
                logger.warning(f"📹 RTSP Grabber: Frame capture failed ({consecutive_fails}/{MAX_FAILS_BEFORE_RECONNECT})")
                if consecutive_fails >= MAX_FAILS_BEFORE_RECONNECT:
                    rtsp_status["state"] = "error"
                    rtsp_status["message"] = "Stream abgebrochen. Reconnecting..."
                    logger.warning("📹 RTSP Grabber: Too many failures. Reconnecting...")
                    cap.release()
                    cap = None
                    current_url = None
                    consecutive_fails = 0
                    time.sleep(5)
                continue
            
            consecutive_fails = 0
            frame_counter += 1
            
            # --- Interval Logic ---
            should_analyze = False
            if interval_mode == "frames":
                # Analyze every Nth frame
                if frame_counter % max(interval_value, 1) == 0:
                    should_analyze = True
                else:
                    # Skip this frame — just grab to keep the stream flowing
                    continue
            else:
                # "seconds" mode: we already sleep at the bottom
                should_analyze = True
            
            if should_analyze:
                # Write frame as JPEG into /events/{camera_name}/
                cam_folder = os.path.join(EVENTS_DIR, camera_name)
                os.makedirs(cam_folder, exist_ok=True)
                
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                frame_filename = f"rtsp_{camera_name}.{ts}.jpg"
                frame_path = os.path.join(cam_folder, frame_filename)
                
                cv2.imwrite(frame_path, frame)
                
                rtsp_status["frames_captured"] += 1
                rtsp_status["last_capture"] = datetime.utcnow().isoformat()
                
                # Calculate FPS (captures per minute for readability)
                now = time.time()
                capture_times.append(now)
                # Keep only last 60 seconds of captures
                capture_times[:] = [t for t in capture_times if now - t < 60]
                if len(capture_times) > 1:
                    elapsed = capture_times[-1] - capture_times[0]
                    rtsp_status["fps"] = round(len(capture_times) / max(elapsed, 1), 2)
                
                mode_label = f"alle {interval_value}s" if interval_mode == "seconds" else f"jeder {interval_value}. Frame"
                rtsp_status["state"] = "connected"
                rtsp_status["message"] = f"Aktiv — {rtsp_status['frames_captured']} Frames — {mode_label}"
                
                logger.info(f"📹 RTSP Grabber: Captured frame #{rtsp_status['frames_captured']} → {frame_filename}")
            
            # Sleep only in seconds mode
            if interval_mode == "seconds":
                time.sleep(interval_value)
            
        except Exception as e:
            logger.error(f"📹 RTSP Grabber error: {e}")
            rtsp_status["state"] = "error"
            rtsp_status["message"] = f"Fehler: {str(e)[:80]}"
            if cap is not None:
                cap.release()
                cap = None
                current_url = None
            time.sleep(10)


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
        plate = result["plate"]
        conf = result["confidence"]
        has_plate = bool(plate and plate != "UNKNOWN" and conf > 0.3)
        logger.info(f"  [{os.path.basename(img_path)}] Detected: {plate} (Conf: {conf:.2f})")
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

def start_watcher():
    if not os.path.exists(EVENTS_DIR):
        os.makedirs(EVENTS_DIR, exist_ok=True)
    
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
    
    # Start RTSP grabber thread
    rtsp_thread = threading.Thread(target=rtsp_grabber, daemon=True)
    rtsp_thread.start()
    
    event_handler = EventFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, EVENTS_DIR, recursive=True)
    observer.start()
    logger.info(f"⚡ Started monitoring {EVENTS_DIR} — Fast-First mode active.")
    return observer

