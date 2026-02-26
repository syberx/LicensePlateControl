from fastapi import APIRouter, Depends, HTTPException, Query
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

