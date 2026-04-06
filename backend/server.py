from fastapi import FastAPI, APIRouter, WebSocket
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import uuid
from datetime import datetime, timezone
import numpy as np
import json
import asyncio
from cjpt_system import cjpt

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Existing models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

# NEW: CJPT Models
class LigoData(BaseModel):
    frequency: List[float]
    strain: List[float]
    snr: float
    detection: bool

# Existing routes
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    return status_checks

# NEW: CJPT Endpoints
@api_router.get("/ligo/generate")
async def generate_ligo_data():
    """Generate LIGO gravitational wave data with CJPT"""
    data = cjpt.generate_ligo_data_full()
    data['timestamp'] = datetime.now(timezone.utc).isoformat()
    return data

@api_router.get("/cjpt/f2scan")
async def f2_scan():
    """Run f2 parameter scan"""
    return cjpt.f2_scan()

@api_router.get("/nanograph/data")
async def get_nanograph():
    """Get nanograph data with physics"""
    return cjpt.generate_nanograph_physics()

@api_router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        for i in range(100):
            data = {"step": i, "value": float(np.random.randn())}
            await websocket.send_json(data)
            await asyncio.sleep(0.1)
    except:
        pass
    finally:
        await websocket.close()

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
