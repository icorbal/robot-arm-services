"""FastAPI REST endpoints for the robot arm services."""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .executor import TaskExecutor

logger = logging.getLogger(__name__)

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
