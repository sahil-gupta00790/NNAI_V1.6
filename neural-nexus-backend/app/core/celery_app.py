# app/core/celery_app.py
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "neural_nexus_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    # Remove "tasks.quantization_tasks" from the include list [1]
    include=["tasks.evolution_tasks"] # Only include evolution tasks
)

# Optional configuration settings
celery_app.conf.update(
    task_track_started=True,
    # Add other Celery configurations as needed
)

