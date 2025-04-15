# app/core/celery_app.py
from celery import Celery
from app.core.config import settings # Assuming settings has REDIS_URL

celery_app = Celery(
    "neural_nexus_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['tasks.evolution_tasks'] # Make sure your task module is included
)

# --- ADD/MODIFY these settings ---
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'], # Only accept JSON
    # Optional: task_track_started=True, # Can be useful for debugging state
    # Optional: broker_connection_retry_on_startup=True, # Good practice
)
# ---------------------------------

# Optional: Load task modules here if not using 'include'
# celery_app.autodiscover_tasks(['tasks'])

