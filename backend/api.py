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
    processing_time_ms: Optional[float] = None

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
    db_plate = db.query(models.Plate).filter(models.Plate.plate_text == plate.plate_text).first()
    if db_plate:
        raise HTTPException(status_code=400, detail="Plate already registered")
    new_plate = models.Plate(plate_text=plate.plate_text, description=plate.description)
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

@router.post("/api/external/plates", response_model=PlateResponse)
def external_push_plate(plate: PlateCreate, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    db_plate = db.query(models.Plate).filter(models.Plate.plate_text == plate.plate_text).first()
    if db_plate:
        db_plate.active = True
        db.commit()
        db.refresh(db_plate)
        return db_plate
    
    new_plate = models.Plate(plate_text=plate.plate_text, description=plate.description)
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
