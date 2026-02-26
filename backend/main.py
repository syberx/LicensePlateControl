from fastapi import FastAPI
import database
import models
import api
import watcher
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy import inspect, text
import logging

logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=database.engine)

# Auto-migrate: add missing columns to existing tables
def _auto_migrate():
    """Add columns defined in models but missing from the database."""
    inspector = inspect(database.engine)
    with database.engine.begin() as conn:
        for table_name, model in [
            ("events", models.Event),
            ("event_images", models.EventImage),
        ]:
            if not inspector.has_table(table_name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table_name)}
            for col in model.__table__.columns:
                if col.name not in existing:
                    col_type = col.type.compile(database.engine.dialect)
                    conn.execute(text(
                        f'ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}'
                    ))
                    logger.info(f"Migration: added {table_name}.{col.name} ({col_type})")

_auto_migrate()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    observer = watcher.start_watcher()
    yield
    # Shutdown
    observer.stop()
    observer.join()

app = FastAPI(title="LicensePlateControl Backend API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)

@app.get("/")
def read_root():
    return {"status": "Backend running", "docs": "/docs"}
