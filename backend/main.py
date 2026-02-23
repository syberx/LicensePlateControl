from fastapi import FastAPI
import database
import models
import api
import watcher
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

models.Base.metadata.create_all(bind=database.engine)

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
