from sqlalchemy import Boolean, Column, Integer, String, DateTime, Float, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from database import Base
import datetime

class Plate(Base):
    __tablename__ = "plates"

    id = Column(Integer, primary_key=True, index=True)
    plate_text = Column(String, unique=True, index=True)
    active = Column(Boolean, default=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.now)

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    detected_plate = Column(String, index=True)
    confidence = Column(String)
    decision = Column(String)
    # Phase 5: Motion burst series tracking
    series_id = Column(String, nullable=True, index=True)
    image_count = Column(Integer, default=1)
    matched_plate = Column(String, nullable=True)
    match_score = Column(Float, nullable=True)
    ha_triggered = Column(Boolean, default=False)
    mqtt_triggered = Column(Boolean, default=False)
    trigger_timestamp = Column(DateTime, nullable=True)
    # Phase 6: Engine diagnostics
    processing_time_ms = Column(Float, nullable=True)
    # Phase 8: Recognition source tracking
    recognition_source = Column(String, nullable=True)  # fast_alpr | paddleocr | vision_llm | hybrid
    vlm_processing_time_ms = Column(Float, nullable=True)  # Fallback OCR time (PaddleOCR or Vision LLM)
    # Phase 7: Relationship to images
    images = relationship("EventImage", back_populates="event", order_by="EventImage.id")

class EventImage(Base):
    __tablename__ = "event_images"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    image_data = Column(LargeBinary, nullable=False)
    detected_plate = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    has_plate = Column(Boolean, default=False)
    is_trigger = Column(Boolean, default=False)  # First plate detection that triggered HA
    created_at = Column(DateTime, default=datetime.datetime.now)
    processing_time_ms = Column(Float, nullable=True)
    recognition_source = Column(String, nullable=True)  # fast_alpr | paddleocr | vision_llm
    plate_crop_data = Column(LargeBinary, nullable=True)  # Cropped plate region from YOLO detection

    event = relationship("Event", back_populates="images")

    @property
    def has_crop(self) -> bool:
        return self.plate_crop_data is not None

class Setting(Base):
    __tablename__ = "settings"

    key = Column(String, primary_key=True, index=True)
    value = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)
