import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOCAL_VLM_MODEL_DIR = BASE_DIR.parent / "models" / "MiniCPM-V-4.6"
DEFAULT_VLM_MODEL = os.environ.get(
    "FALL_GATEWAY_VLM_MODEL",
    str(LOCAL_VLM_MODEL_DIR)
    if LOCAL_VLM_MODEL_DIR.exists()
    else "openbmb/MiniCPM-V-4.6",
)

DATA_DIR = BASE_DIR / "data"
TEST_VIDEO_DIR = DATA_DIR / "test_videos"
EVENT_DIR = DATA_DIR / "events"
DB_PATH = DATA_DIR / "records.db"
