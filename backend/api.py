from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from typing import List, Optional, Any
from database import get_db
import models
from pydantic import BaseModel, ConfigDict
import datetime
import os
from fastapi.security.api_key import APIKeyHeader
from fastapi import Security

router = APIRouter()

API_KEY = os.getenv("API_KEY", "secretmvpkey123")
EVENTS_DIR = "/events"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

class PlateCreate(BaseModel):
    plate_text: str
    description: Optional[str] = None

class PlateResponse(BaseModel):
    id: int
    plate_text: str
    active: bool
    description: Optional[str] = None
    created_at: datetime.datetime
    # Phase 6: Stats
    detect_count: Optional[int] = 0
    last_seen: Optional[datetime.datetime] = None

    model_config = ConfigDict(from_attributes=True)

class ActiveUpdate(BaseModel):
    active: bool

class EventImageResponse(BaseModel):
    id: int
    filename: str
    detected_plate: Optional[str] = None
    confidence: Optional[float] = None
    has_plate: bool = False
    is_trigger: bool = False
    created_at: Optional[datetime.datetime] = None
    processing_time_ms: Optional[float] = None
    recognition_source: Optional[str] = None
    has_crop: bool = False

    model_config = ConfigDict(from_attributes=True)

class EventResponse(BaseModel):
    id: int
    timestamp: datetime.datetime
    detected_plate: Optional[str] = None
    confidence: Optional[str] = None
    decision: Optional[str] = None
    series_id: Optional[str] = None
    image_count: Optional[int] = 1
    matched_plate: Optional[str] = None
    matched_plate_description: Optional[str] = None
    match_score: Optional[float] = None
    ha_triggered: Optional[bool] = False
    mqtt_triggered: Optional[bool] = False
    trigger_timestamp: Optional[datetime.datetime] = None
    processing_time_ms: Optional[float] = None
    recognition_source: Optional[str] = None
    vlm_processing_time_ms: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)

class SettingCreate(BaseModel):
    key: str
    value: str

class SettingResponse(BaseModel):
    key: str
    value: str

    model_config = ConfigDict(from_attributes=True)

@router.post("/api/plates", response_model=PlateResponse)
def create_plate(plate: PlateCreate, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    from watcher import apply_corrections, validate_plate
    # Normalize plate text (auto-insert dashes, handle E/H suffix)
    normalized = apply_corrections(plate.plate_text)
    if not validate_plate(normalized):
        raise HTTPException(status_code=400, detail=f"Ungültiges Format: '{plate.plate_text}' -> '{normalized}'. Erwartet: XX-YY-1234 (optional E/H)")
    db_plate = db.query(models.Plate).filter(models.Plate.plate_text == normalized).first()
    if db_plate:
        raise HTTPException(status_code=400, detail="Kennzeichen bereits registriert")
    new_plate = models.Plate(plate_text=normalized, description=plate.description)
    db.add(new_plate)
    db.commit()
    db.refresh(new_plate)
    return new_plate

@router.get("/api/plates", response_model=List[PlateResponse])
def get_plates(db: Session = Depends(get_db)):
    plates = db.query(models.Plate).all()
    # Calculate stats per plate
    res = []
    for p in plates:
        count = db.query(func.count(models.Event.id)).filter(models.Event.matched_plate == p.plate_text).scalar()
        last = db.query(func.max(models.Event.timestamp)).filter(models.Event.matched_plate == p.plate_text).scalar()
        
        pr = PlateResponse.model_validate(p)
        pr.detect_count = count or 0
        pr.last_seen = last
        res.append(pr)
    return res

@router.patch("/api/plates/{plate_id}/active", response_model=PlateResponse)
def update_plate_active(plate_id: int, update: ActiveUpdate, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    db_plate = db.query(models.Plate).filter(models.Plate.id == plate_id).first()
    if not db_plate:
        raise HTTPException(status_code=404, detail="Plate not found")
    db_plate.active = update.active
    db.commit()
    db.refresh(db_plate)
    return db_plate

@router.delete("/api/plates/{plate_id}")
def delete_plate(plate_id: int, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    db_plate = db.query(models.Plate).filter(models.Plate.id == plate_id).first()
    if not db_plate:
        raise HTTPException(status_code=404, detail="Plate not found")
    db.delete(db_plate)
    db.commit()
    return {"status": "ok", "message": "Kennzeichen gelöscht"}

@router.post("/api/external/plates", response_model=PlateResponse)
def external_push_plate(plate: PlateCreate, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    from watcher import apply_corrections, validate_plate
    normalized = apply_corrections(plate.plate_text)
    if not validate_plate(normalized):
        raise HTTPException(status_code=400, detail=f"Ungültiges Format: '{plate.plate_text}'. Erwartet: XX-YY-1234 (optional E/H)")
    db_plate = db.query(models.Plate).filter(models.Plate.plate_text == normalized).first()
    if db_plate:
        db_plate.active = True
        db.commit()
        db.refresh(db_plate)
        return db_plate

    new_plate = models.Plate(plate_text=normalized, description=plate.description)
    db.add(new_plate)
    db.commit()
    db.refresh(new_plate)
    return new_plate

@router.get("/api/events", response_model=List[EventResponse])
def get_events(limit: int = 100, hide_unknown: bool = False, db: Session = Depends(get_db)):
    query = db.query(models.Event)
    if hide_unknown:
        query = query.filter(models.Event.detected_plate != "UNKNOWN")
    events = query.order_by(models.Event.timestamp.desc()).limit(limit).all()
    
    # Enrich with matched plate description
    result = []
    for event in events:
        er = EventResponse.model_validate(event)
        if event.matched_plate:
            plate = db.query(models.Plate).filter(models.Plate.plate_text == event.matched_plate).first()
            if plate:
                er.matched_plate_description = plate.description
        result.append(er)
    return result

@router.get("/api/events/{event_id}", response_model=EventResponse)
def get_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(models.Event).filter(models.Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    er = EventResponse.model_validate(event)
    if event.matched_plate:
        plate = db.query(models.Plate).filter(models.Plate.plate_text == event.matched_plate).first()
        if plate:
            er.matched_plate_description = plate.description
    return er

@router.delete("/api/events")
def clear_all_events(db: Session = Depends(get_db)):
    """Delete all events and associated images from the database."""
    try:
        # Cascade delete is configured on the model, but let's be safe
        db.query(models.EventImage).delete()
        db.query(models.Event).delete()
        db.commit()
        return {"status": "ok", "message": "Alle Events und Bilder gelöscht"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/events/{event_id}/images", response_model=List[EventImageResponse])
def get_event_images(event_id: int, db: Session = Depends(get_db)):
    """Return metadata for all images in an event (without binary data)."""
    images = db.query(models.EventImage).filter(
        models.EventImage.event_id == event_id
    ).order_by(models.EventImage.id).all()
    return images

@router.get("/api/images/db/{image_id}")
def get_image_from_db(image_id: int, db: Session = Depends(get_db)):
    """Serve an image from the database by EventImage ID."""
    img = db.query(models.EventImage).filter(models.EventImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Determine media type from filename
    ext = img.filename.lower().split('.')[-1]
    media_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
    
    return Response(content=img.image_data, media_type=media_type)

@router.get("/api/images/db/{image_id}/crop")
def get_crop_from_db(image_id: int, db: Session = Depends(get_db)):
    """Serve the cropped plate region for an EventImage."""
    img = db.query(models.EventImage).filter(models.EventImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    if not img.plate_crop_data:
        raise HTTPException(status_code=404, detail="No plate crop available")
    return Response(content=img.plate_crop_data, media_type="image/jpeg")

@router.get("/api/images/{file_path:path}")
def get_image(file_path: str):
    """Serve event images from /events/ directory (legacy fallback)."""
    full_path = os.path.join(EVENTS_DIR, file_path)
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="Image not found")
    # Security: ensure path is within EVENTS_DIR
    if not os.path.abspath(full_path).startswith(os.path.abspath(EVENTS_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(full_path)

@router.get("/api/settings", response_model=List[SettingResponse])
def get_settings(db: Session = Depends(get_db)):
    return db.query(models.Setting).all()

@router.post("/api/settings", response_model=SettingResponse)
def update_setting(setting: SettingCreate, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    db_setting = db.query(models.Setting).filter(models.Setting.key == setting.key).first()
    if db_setting:
        db_setting.value = setting.value
    else:
        db_setting = models.Setting(key=setting.key, value=setting.value)
        db.add(db_setting)
    db.commit()
    db.refresh(db_setting)
    return db_setting

@router.get("/api/rtsp/status")
def get_rtsp_status():
    """Return current RTSP grabber status (state, frames, detections, etc.)."""
    from watcher import rtsp_status
    return rtsp_status

@router.get("/api/rtsp/preview")
def get_rtsp_preview():
    """Serve the latest RTSP frame as a JPEG image for the live preview."""
    from watcher import rtsp_preview_frame
    if rtsp_preview_frame is None:
        raise HTTPException(status_code=404, detail="No preview available")
    return Response(content=rtsp_preview_frame, media_type="image/jpeg")

@router.get("/api/rtsp/preview_unmasked")
def get_rtsp_preview_unmasked():
    """Serve the latest RTSP frame BEFORE the ROI mask is applied, for drawing."""
    from watcher import rtsp_unmasked_preview_frame
    if rtsp_unmasked_preview_frame is None:
        raise HTTPException(status_code=404, detail="No unmasked preview available")
    return Response(content=rtsp_unmasked_preview_frame, media_type="image/jpeg")


# --- System Logs ---

@router.get("/api/logs")
def get_logs(level: str = None, source: str = None, limit: int = 200):
    """Return recent system log entries from the in-memory ring buffer."""
    from watcher import log_handler
    return log_handler.get_entries(level=level, source=source, limit=limit)

@router.delete("/api/logs")
def clear_logs():
    """Clear the in-memory log buffer."""
    from watcher import log_handler
    log_handler.entries.clear()
    return {"status": "ok", "message": "Logs cleared"}

# --- Debug Frame Buffer ---

@router.get("/api/debug/frames")
def get_debug_frames():
    """Return metadata of all frames in the debug buffer (without image data)."""
    from watcher import debug_frame_buffer
    return [
        {
            "id": e["id"],
            "timestamp": e["timestamp"],
            "plate": e["plate"],
            "confidence": e["confidence"],
            "processing_ms": e["processing_ms"],
            "has_plate": bool(e["plate"] and e["plate"] != "UNKNOWN" and e["confidence"] > 0.3),
        }
        for e in debug_frame_buffer
    ]

@router.get("/api/debug/frames/{frame_id}")
def get_debug_frame_image(frame_id: int):
    """Serve a single debug frame as JPEG by its ID."""
    from watcher import debug_frame_buffer
    for entry in debug_frame_buffer:
        if entry["id"] == frame_id:
            return Response(content=entry["jpeg"], media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Frame not found")

@router.delete("/api/debug/frames")
def clear_debug_frames():
    """Clear the debug frame buffer."""
    from watcher import debug_frame_buffer
    debug_frame_buffer.clear()
    return {"status": "ok", "message": "Debug buffer cleared"}

@router.get("/api/debug/stats")
def get_debug_stats():
    """Return debug buffer statistics."""
    from watcher import debug_frame_buffer, DEBUG_BUFFER_MAX_AGE_SECONDS
    count = len(debug_frame_buffer)
    size_mb = sum(len(e["jpeg"]) for e in debug_frame_buffer) / (1024 * 1024) if count else 0
    plates = sum(1 for e in debug_frame_buffer if e["plate"] and e["plate"] != "UNKNOWN" and e["confidence"] > 0.3)
    oldest = debug_frame_buffer[0]["timestamp"] if count else None
    newest = debug_frame_buffer[-1]["timestamp"] if count else None
    return {
        "frame_count": count,
        "size_mb": round(size_mb, 1),
        "detections": plates,
        "oldest": oldest,
        "newest": newest,
        "max_age_seconds": DEBUG_BUFFER_MAX_AGE_SECONDS,
    }


# --- Debug Pipeline: Step-by-step RTSP simulation ---

@router.post("/api/debug/pipeline")
async def debug_pipeline(file: UploadFile = File(...), detect_width: int = 320, preprocess: bool = True):
    """Simulate the full RTSP 2-pass pipeline on an uploaded image.
    Returns step-by-step results with intermediate images (base64), timings, and metadata.
    detect_width: YOLO detection resolution (320/416/640). Lower = faster but less accurate.
    preprocess: Apply CLAHE+Schärfen auf den Crop vor OCR."""
    import cv2
    import numpy as np
    import base64
    import time
    import json
    import requests
    from watcher import (
        _get_cached_polygon, get_setting, apply_corrections,
        ENGINE_URL, _detect_plates, _ocr_plate, _analyze_frame_bytes,
    )
    from database import SessionLocal

    detect_width = max(160, min(detect_width, 640))  # clamp 160..640

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Ungültiges Bild")

    steps = []
    total_start = time.time()

    # --- Step 1: Original Image ---
    t0 = time.time()
    orig_h, orig_w = frame.shape[:2]
    _, orig_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    orig_b64 = base64.b64encode(orig_buf.tobytes()).decode('ascii')
    steps.append({
        "step": 1,
        "name": "Original-Bild",
        "description": f"Eingelesenes Bild: {orig_w}x{orig_h}px ({len(contents)/1024:.0f} KB)",
        "image_b64": orig_b64,
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "details": {"width": orig_w, "height": orig_h, "size_kb": round(len(contents)/1024, 1)},
    })

    # --- Step 2: ROI Crop (Bbox only, kein polygon-bitwise_and für Detection) ---
    # Für die Detection-Phase brauchen wir nur die Position, nicht pixelgenaue Maskierung.
    # Bbox-Crop reicht völlig aus und ist 10-50x schneller als polygon-bitwise_and.
    t0 = time.time()
    db = SessionLocal()
    polygon_str = get_setting(db, "rtsp_roi_polygon", "")
    db.close()

    has_roi = False
    x_rect, y_rect, w_rect, h_rect = 0, 0, orig_w, orig_h
    frame_for_engine = frame

    if polygon_str:
        try:
            points = json.loads(polygon_str)
            if len(points) >= 3:
                h, w = frame.shape[:2]
                poly_pts = np.array([
                    [int(p["x"] * w), int(p["y"] * h)] for p in points
                ], np.int32).reshape((-1, 1, 2))

                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(poly_pts)
                x_rect = max(0, x_rect)
                y_rect = max(0, y_rect)
                w_rect = min(w - x_rect, w_rect)
                h_rect = min(h - y_rect, h_rect)

                # Nur Bbox-Crop, kein polygon-Fill/bitwise_and — reicht für YOLO
                frame_for_engine = frame[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect].copy()
                has_roi = True

                _, crop_buf = cv2.imencode('.jpg', frame_for_engine, [cv2.IMWRITE_JPEG_QUALITY, 75])
                crop_b64_roi = base64.b64encode(crop_buf.tobytes()).decode('ascii')
                mh, mw = frame_for_engine.shape[:2]
                steps.append({
                    "step": 2,
                    "name": "ROI Bbox-Crop",
                    "description": f"ROI Bbox-Crop ({len(points)} Punkte) → {mw}x{mh}px (Offset: {x_rect},{y_rect}) — kein polygon-Fill für Detection nötig",
                    "image_b64": crop_b64_roi,
                    "duration_ms": round((time.time() - t0) * 1000, 1),
                    "details": {"roi_points": len(points), "crop_w": mw, "crop_h": mh, "offset_x": x_rect, "offset_y": y_rect, "mode": "bbox_crop_only"},
                })
        except Exception as e:
            steps.append({
                "step": 2, "name": "ROI Bbox-Crop",
                "description": f"ROI-Fehler: {e}", "image_b64": None,
                "duration_ms": round((time.time() - t0) * 1000, 1), "details": {"error": str(e)},
            })
    else:
        steps.append({
            "step": 2, "name": "ROI Bbox-Crop",
            "description": "Kein ROI konfiguriert — volles Bild wird verwendet",
            "image_b64": None, "duration_ms": round((time.time() - t0) * 1000, 1),
            "details": {"roi_active": False},
        })

    # --- Step 3: Resize für YOLO (detect_width konfigurierbar, Standard 416) ---
    # Für Detection reicht kleine Auflösung — YOLO findet Positionen auch bei 320px.
    # Niedrigere Auflösung = viel schneller. OCR läuft erst bei Treffer auf Hi-Res-Crop.
    t0 = time.time()
    eh, ew = frame_for_engine.shape[:2]
    resize_scale = 1.0
    if ew > detect_width:
        resize_scale = detect_width / ew
        new_w = detect_width
        new_h = int(eh * resize_scale)
        frame_for_engine = cv2.resize(frame_for_engine, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = ew, eh

    _, resize_buf = cv2.imencode('.jpg', frame_for_engine, [cv2.IMWRITE_JPEG_QUALITY, 72])
    resize_b64 = base64.b64encode(resize_buf.tobytes()).decode('ascii')
    engine_bytes = resize_buf.tobytes()

    steps.append({
        "step": 3, "name": f"Resize für YOLO ({detect_width}px)",
        "description": f"{ew}x{eh} → {new_w}x{new_h}px — Nur Positionsfindung, Qualität egal",
        "image_b64": resize_b64,
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "details": {"original_w": ew, "original_h": eh, "resized_w": new_w, "resized_h": new_h, "scale": round(resize_scale, 4), "detect_width": detect_width},
    })

    # --- Step 4: /detect (YOLO Detection) ---
    t0 = time.time()
    detect_result = _detect_plates(engine_bytes, "debug_pipeline.jpg")
    detect_ms = detect_result.get("processing_time_ms")
    detections = detect_result.get("detections", [])
    detect_mode = detect_result.get("mode", "unknown")

    # Draw bboxes on the 640px image for visualization
    det_vis = frame_for_engine.copy()
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) == 4:
            cv2.rectangle(det_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(det_vis, f"{det.get('confidence', 0):.2f}", (bbox[0], bbox[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    _, det_buf = cv2.imencode('.jpg', det_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    det_b64 = base64.b64encode(det_buf.tobytes()).decode('ascii')

    steps.append({
        "step": 4, "name": "/detect (YOLO)",
        "description": f"{len(detections)} Kennzeichen gefunden (Engine: {detect_ms:.0f}ms, Modus: {detect_mode})" if detections else f"Keine Kennzeichen gefunden ({detect_ms or 0:.0f}ms, Modus: {detect_mode})",
        "image_b64": det_b64,
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "details": {"detections": detections, "engine_ms": detect_ms, "mode": detect_mode},
    })

    # --- Step 5: BBox-Mapping + Hi-Res Crop ---
    plate = ""
    confidence = 0.0
    crop_b64 = None

    if detections:
        t0 = time.time()
        best_det = max(detections, key=lambda d: d.get("confidence", 0))
        engine_bbox = best_det.get("bbox", [])

        if len(engine_bbox) == 4:
            ex1, ey1, ex2, ey2 = engine_bbox
            inv_scale = 1.0 / resize_scale if resize_scale != 1.0 else 1.0
            rx1 = int(ex1 * inv_scale)
            ry1 = int(ey1 * inv_scale)
            rx2 = int(ex2 * inv_scale)
            ry2 = int(ey2 * inv_scale)
            if has_roi:
                ox1 = rx1 + x_rect
                oy1 = ry1 + y_rect
                ox2 = rx2 + x_rect
                oy2 = ry2 + y_rect
            else:
                ox1, oy1, ox2, oy2 = rx1, ry1, rx2, ry2
            fh, fw = frame.shape[:2]
            # Padding: 15% horizontal, 20% vertikal — damit keine Ziffer abgeschnitten wird
            pad_x = max(6, int((ox2 - ox1) * 0.10))
            pad_y = max(4, int((oy2 - oy1) * 0.08))
            ox1, oy1 = max(0, ox1 - pad_x), max(0, oy1 - pad_y)
            ox2, oy2 = min(fw, ox2 + pad_x), min(fh, oy2 + pad_y)

            # Visualization: auf max 960px runterskalieren vor JPEG-Encode (spart 200-400ms bei 4K)
            vis_max_w = 960
            vis_scale = min(1.0, vis_max_w / orig_w)
            if vis_scale < 1.0:
                vis_frame = cv2.resize(frame, (int(orig_w * vis_scale), int(orig_h * vis_scale)), interpolation=cv2.INTER_AREA)
                vx1, vy1 = int(ox1 * vis_scale), int(oy1 * vis_scale)
                vx2, vy2 = int(ox2 * vis_scale), int(oy2 * vis_scale)
            else:
                vis_frame = frame.copy()
                vx1, vy1, vx2, vy2 = ox1, oy1, ox2, oy2
            cv2.rectangle(vis_frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Crop: {ox2-ox1}x{oy2-oy1}px", (vx1, max(0, vy1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            _, bbox_vis_buf = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            bbox_vis_b64 = base64.b64encode(bbox_vis_buf.tobytes()).decode('ascii')

            hires_crop = frame[oy1:oy2, ox1:ox2]
            crop_h, crop_w = hires_crop.shape[:2] if hires_crop.size > 0 else (0, 0)

            if hires_crop.size > 0:
                _, crop_buf = cv2.imencode('.jpg', hires_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                from watcher import _preprocess_crop_for_ocr
                raw_crop_bytes = crop_buf.tobytes()
                # Auto-Preprocessing: immer AN wenn Crop zu klein (pixelig)
                crop_h_px = oy2 - oy1
                auto_preprocess = preprocess or (crop_h_px < 60)
                if auto_preprocess:
                    crop_jpeg_bytes = _preprocess_crop_for_ocr(raw_crop_bytes)
                else:
                    crop_jpeg_bytes = raw_crop_bytes
                # Show the crop that OCR actually receives
                crop_b64 = base64.b64encode(crop_jpeg_bytes).decode('ascii')

            prep_label = " + Preprocessing (CLAHE+Schärfen)" if (preprocess or (oy2 - oy1) < 60) else " — Preprocessing AUS"
            steps.append({
                "step": 5, "name": "BBox-Mapping + Hi-Res Crop",
                "description": f"Engine-BBox {engine_bbox} → Original [{ox1},{oy1},{ox2},{oy2}] → Crop: {crop_w}x{crop_h}px{prep_label}",
                "image_b64": bbox_vis_b64,
                "duration_ms": round((time.time() - t0) * 1000, 1),
                "details": {
                    "engine_bbox": engine_bbox, "inv_scale": round(inv_scale, 4),
                    "roi_offset": [x_rect, y_rect] if has_roi else [0, 0],
                    "original_bbox": [ox1, oy1, ox2, oy2], "crop_size": [crop_w, crop_h],
                },
            })

            # --- Step 6a: Fast ALPR OCR (immer) ---
            if crop_b64 and hires_crop.size > 0:
                t0 = time.time()
                ocr_result = _ocr_plate(crop_jpeg_bytes, "debug_crop.jpg")
                plate = ocr_result.get("plate", "")
                confidence = ocr_result.get("confidence", 0.0)
                ocr_ms = ocr_result.get("processing_time_ms")
                ocr_mode = ocr_result.get("mode", "unknown")

                steps.append({
                    "step": 6, "name": "Fast ALPR OCR",
                    "description": f"Erkannt: '{plate}' (Conf: {confidence:.2f}, {ocr_ms or 0:.0f}ms)" if plate and plate != "UNKNOWN" else f"Nicht erkannt ({ocr_ms or 0:.0f}ms)",
                    "image_b64": crop_b64,
                    "duration_ms": round((time.time() - t0) * 1000, 1),
                    "details": {"plate": plate, "confidence": confidence, "engine_ms": ocr_ms, "mode": ocr_mode},
                })

    else:
        steps.append({
            "step": 5, "name": "BBox-Mapping + Hi-Res Crop",
            "description": "Uebersprungen — keine Detection", "image_b64": None,
            "duration_ms": 0, "details": {},
        })

    # --- Step 7 (optional): Fallback /analyze nur wenn Detection war, aber OCR fehlgeschlagen ---
    # Kein Fallback wenn YOLO gar nichts gefunden hat — dann ist das Bild leer, /analyze unnötig
    if detections and (not plate or plate == "UNKNOWN"):
        t0 = time.time()
        fallback_result = _analyze_frame_bytes(engine_bytes, "debug_fallback.jpg")
        f_plate = fallback_result.get("plate", "")
        f_conf = fallback_result.get("confidence", 0.0)
        f_ms = fallback_result.get("processing_time_ms")
        steps.append({
            "step": 7, "name": "Fallback: /analyze (Single-Pass)",
            "description": f"Erkannt: '{f_plate}' (Conf: {f_conf:.2f}, {f_ms:.0f}ms)" if f_plate and f_plate != "UNKNOWN" else f"Auch nichts erkannt ({f_ms or 0:.0f}ms)",
            "image_b64": None,
            "duration_ms": round((time.time() - t0) * 1000, 1),
            "details": {"plate": f_plate, "confidence": f_conf, "engine_ms": f_ms},
        })
        if f_plate and f_plate != "UNKNOWN" and f_conf > 0:
            plate = f_plate
            confidence = f_conf

    total_ms = round((time.time() - total_start) * 1000, 1)

    return {
        "steps": steps,
        "total_duration_ms": total_ms,
        "final_plate": plate,
        "final_confidence": confidence,
        "step_count": len(steps),
    }
