"""FastAPI REST endpoints for the robot arm services."""

import logging
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .executor import TaskExecutor

logger = logging.getLogger(__name__)


def _get_git_version() -> dict[str, str]:
    """Read git commit hash and timestamp at startup."""
    repo_dir = Path(__file__).parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        timestamp = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"],
            cwd=repo_dir, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=repo_dir, stderr=subprocess.DEVNULL,
        ) != 0
        return {
            "service": "raserv",
            "commit": commit + ("-dirty" if dirty else ""),
            "committed": timestamp,
        }
    except Exception:
        return {"service": "raserv", "commit": "unknown", "committed": "unknown"}


_VERSION = _get_git_version()

app = FastAPI(
    title="Robot Arm Services (RAServ)",
    description="LLM-powered task planner and executor for robot arm control",
    version="1.0.0",
)

# Global executor instance (set by run.py)
_executor: TaskExecutor | None = None


def set_executor(executor: TaskExecutor) -> None:
    """Set the global executor instance."""
    global _executor
    _executor = executor


def get_executor() -> TaskExecutor:
    """Get the global executor instance."""
    if _executor is None:
        raise HTTPException(status_code=503, detail="Executor not initialized")
    return _executor


# --- Request/Response Models ---

class TaskRequest(BaseModel):
    prompt: str = Field(..., description="Natural language task description")
    max_iterations: int | None = Field(None, description="Override max iterations")


class HealthResponse(BaseModel):
    status: str = "ok"


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/version")
async def version():
    """Return service version (git commit)."""
    return _VERSION


@app.post("/task")
async def execute_task(request: TaskRequest):
    """Execute a natural language task.

    Sends the task through the plan-execute-verify loop:
    1. Gets scene state from RASim
    2. Plans next step via LLM
    3. Executes commands on RASim
    4. Verifies completion via LLM
    5. Loops until done or max iterations
    """
    executor = get_executor()
    logger.info(f"Received task: {request.prompt}")

    try:
        result = await executor.execute_task(request.prompt)
        return result
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/task/cancel")
async def cancel_task():
    """Cancel the currently running task."""
    executor = get_executor()
    cancelled = executor.cancel()
    if cancelled:
        return {"status": "cancelled", "message": "Task cancellation requested"}
    return {"status": "idle", "message": "No task is currently running"}


@app.get("/task/status")
async def task_status():
    """Check if a task is currently running."""
    executor = get_executor()
    return {"running": executor.is_running}
