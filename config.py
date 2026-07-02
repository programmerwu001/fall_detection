import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOCAL_VLM_MODEL_DIR = BASE_DIR / "models" / "MiniCPM-V-4.6"
DEFAULT_VLM_MODEL = os.environ.get(
    "FALL_GATEWAY_VLM_MODEL",
    str(LOCAL_VLM_MODEL_DIR)
    if LOCAL_VLM_MODEL_DIR.exists()
    else "openbmb/MiniCPM-V-4.6",
)

DATA_DIR = BASE_DIR / "data"
TEST_VIDEO_DIR = DATA_DIR / "test_videos"
EVENT_DIR = DATA_DIR / "events"
PRIVATE_EVENT_DIR = DATA_DIR / "private_events"
PRIVACY_PREVIEW_DIR = DATA_DIR / "privacy_previews"
DISABLED_DEBUG_EVENT_DIR = DATA_DIR / "debug_events_disabled"
DB_PATH = DATA_DIR / "records.db"

LOCAL_PRIVACY_PREVIEW_MODEL = BASE_DIR / "models" / "yolo11n-seg.pt"
DEFAULT_PRIVACY_PREVIEW_MODEL = os.environ.get(
    "FALL_GATEWAY_PRIVACY_PREVIEW_MODEL",
    str(LOCAL_PRIVACY_PREVIEW_MODEL)
    if LOCAL_PRIVACY_PREVIEW_MODEL.exists()
    else "yolo11n-seg.pt",
)

DEFAULT_SAVE_DEBUG_RAW_EVENT_COPY = True
DEFAULT_HIGH_RISK_REPEAT_SECONDS = 20
DEFAULT_LOW_RISK_REPEAT_SECONDS = 60
