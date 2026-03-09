"""Plan-execute-verify loop - orchestrates the full task execution pipeline."""

import asyncio
import logging
from typing import Any

import httpx

from .planner import TaskPlanner
from .verifier import TaskVerifier

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Executes tasks by planning, executing on RASim, and verifying in a loop."""

    def __init__(
        self,
        planner: TaskPlanner,
        verifier: TaskVerifier,
        rasim_url: str = "http://localhost:8100",
        max_iterations: int = 10,
        step_delay: float = 0.5,
    ):
        self._planner = planner
        self._verifier = verifier
        self._rasim_url = rasim_url.rstrip("/")
        self._max_iterations = max_iterations
        self._step_delay = step_delay
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info(
            f"TaskExecutor initialized (rasim={rasim_url}, "
            f"max_iter={max_iterations}, delay={step_delay}s)"
        )

    async def close(self) -> None:
        """Clean up HTTP client."""
        await self._client.aclose()

    async def _get_scene_state(self) -> dict[str, Any]:
        """Fetch current scene state from RASim."""
        response = await self._client.get(f"{self._rasim_url}/scene-state")
        response.raise_for_status()
        return response.json()

    async def _execute_commands(self, commands: list[dict]) -> dict[str, Any]:
        """Send commands to RASim for execution."""
        response = await self._client.post(
            f"{self._rasim_url}/execute",
            json={"commands": commands},
        )
        response.raise_for_status()
        return response.json()

    async def execute_task(self, task: str) -> dict[str, Any]:
        """Execute a task through the plan-execute-verify loop.

        Args:
            task: Natural language task description

        Returns:
            Execution log with results
        """
        logger.info(f"Starting task execution: {task}")
        execution_log: list[dict] = []

        for iteration in range(1, self._max_iterations + 1):
            logger.info(f"--- Iteration {iteration}/{self._max_iterations} ---")

            # 1. Get scene state
            try:
                scene_state = await self._get_scene_state()
            except httpx.HTTPError as e:
                error_msg = f"Failed to get scene state: {e}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "task": task,
                    "iterations": iteration,
                    "log": execution_log,
                }

            # 2. Plan next step (with history of previous steps)
            try:
                plan = await self._planner.plan_next_step(scene_state, task, history=execution_log)
            except Exception as e:
                error_msg = f"Planning failed: {e}"
                logger.error(error_msg)
                execution_log.append({
                    "iteration": iteration,
                    "phase": "planning",
                    "error": error_msg,
                })
                return {
                    "status": "error",
                    "error": error_msg,
                    "task": task,
                    "iterations": iteration,
                    "log": execution_log,
                }

            commands = plan.get("commands", [])
            step_desc = plan.get("step_description", "")

            # No commands = planner thinks we're done — verify before stopping
            if not commands:
                logger.info("Planner returned no commands — verifying before stopping")
                try:
                    check = await self._verifier.verify(scene_state, task)
                except Exception as e:
                    check = {"completed": False, "reason": str(e), "confidence": 0.0}

                if check.get("completed", False):
                    logger.info(f"Task verified complete: {check.get('reason')}")
                    return {
                        "status": "completed",
                        "task": task,
                        "iterations": iteration,
                        "final_scene_state": scene_state,
                        "verification": check,
                        "log": execution_log,
                    }

                # Not actually done — tell planner to try again
                logger.warning("Planner thinks done but verifier disagrees — retrying")
                execution_log.append({
                    "iteration": iteration,
                    "phase": "replanning",
                    "step_description": step_desc,
                    "commands": [],
                    "note": f"Planner returned no commands but task not complete: {check.get('reason')}",
                })
                continue

            # 3. Execute commands on RASim
            try:
                exec_result = await self._execute_commands(commands)
            except httpx.HTTPError as e:
                error_msg = f"Command execution failed: {e}"
                logger.error(error_msg)
                execution_log.append({
                    "iteration": iteration,
                    "phase": "execution",
                    "step_description": step_desc,
                    "commands": commands,
                    "error": error_msg,
                })
                return {
                    "status": "error",
                    "error": error_msg,
                    "task": task,
                    "iterations": iteration,
                    "log": execution_log,
                }

            execution_log.append({
                "iteration": iteration,
                "phase": "executed",
                "step_description": step_desc,
                "commands": commands,
                "results": exec_result.get("results", []),
            })

            # 4. Get updated scene state
            try:
                updated_state = await self._get_scene_state()
            except httpx.HTTPError as e:
                logger.error(f"Failed to get updated scene state: {e}")
                updated_state = exec_result.get("scene_state", scene_state)

            # 5. Verify task completion
            try:
                verification = await self._verifier.verify(updated_state, task)
            except Exception as e:
                logger.warning(f"Verification failed: {e}")
                verification = {"completed": False, "reason": f"Verification error: {e}", "confidence": 0.0}

            execution_log[-1]["verification"] = verification

            if verification.get("completed", False):
                logger.info(f"Task completed! Reason: {verification.get('reason')}")
                return {
                    "status": "completed",
                    "task": task,
                    "iterations": iteration,
                    "final_scene_state": updated_state,
                    "verification": verification,
                    "log": execution_log,
                }

            # Delay before next iteration
            if self._step_delay > 0:
                await asyncio.sleep(self._step_delay)

        # Max iterations reached
        logger.warning(f"Max iterations ({self._max_iterations}) reached without task completion")
        final_state = await self._get_scene_state()
        return {
            "status": "max_iterations_reached",
            "task": task,
            "iterations": self._max_iterations,
            "final_scene_state": final_state,
            "log": execution_log,
        }
