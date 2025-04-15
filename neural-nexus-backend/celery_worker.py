# celery_worker.py (at the root level)
from app.main import app # Ensure context is loaded if needed by tasks
from app.core.celery_app import celery_app

# This script is run to start the Celery worker process
# Example command: celery -A celery_worker.celery_app worker --loglevel=info
