"""Configuration management for Phase 1 scripts"""
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_params():
    """Load parameters from params.yaml"""
    with open(PROJECT_ROOT / "params.yaml", 'r') as f:
        return yaml.safe_load(f)

PARAMS = load_params()
#MONGO_URI = os.getenv('MONGO_URI', 'mongodb://admin:password123@localhost:27017')
#MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
