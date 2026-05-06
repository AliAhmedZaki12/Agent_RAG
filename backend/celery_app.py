"""
Celery Application + Tasks
"""
from celery import Celery
from backend.config import REDIS_URL

celery = Celery(
    "agentic_rag",
    broker  = REDIS_URL,
    backend = REDIS_URL,
)

celery.conf.update(
    task_serializer   = "json",
    result_serializer = "json",
    accept_content    = ["json"],
    timezone          = "UTC",
    task_track_started = True,
)


# ── Tasks ────────────────────────────────────────────────────────

@celery.task(bind=True, max_retries=2, default_retry_delay=5)
def run_query(self, question: str):
    try:
        from backend.orchestrator import run
        answer, trace = run(question)
        return {"answer": answer, "trace": trace}
    except Exception as exc:
        raise self.retry(exc=exc)


@celery.task
def run_ingest(texts: list, namespace: str = ""):
    from backend.ingestion import ingest
    count = ingest(texts, namespace=namespace)
    return {"ingested": count}
