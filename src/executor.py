"""Plan-execute-verify loop - orchestrates the full task execution pipeline."""

import asyncio
import logging
from typing import Any

import httpx

from .perception import StereoPerceiver
from .planner import TaskPlanner
from .verifier import TaskVerifier

logger = logging.getLogger(__name__)


class InteractionZoneViolation(Exception):
    """Raised when objects leave the interaction zone during execution."""

    def __init__(self, violations: list[dict]):
        self.violations = violations
        ids = [v["id"] for v in violations]
        super().__init__(f"Objects outside interaction zone: {', '.join(ids)}")


class TaskExecutor:
    """Executes tasks by planning, executing on RASim, and verifying in a loop."""

    def __init__(
        self,
        planner: TaskPlanner,
        verifier: TaskVerifier,
        rasim_url: str = "http://localhost:8100",
        max_iterations: int = 10,
        step_delay: float = 0.5,
        perception_mode: str = "scene_state",
        perceiver: StereoPerceiver | None = None,
    ):
        self._planner = planner
        self._verifier = verifier
        self._rasim_url = rasim_url.rstrip("/")
        self._max_iterations = max_iterations
        self._step_delay = step_delay
        self._perception_mode = perception_mode
        self._perceiver = perceiver
        self._client = httpx.AsyncClient(timeout=10.0)
        self._cancel_event: asyncio.Event | None = None
        self._running = False
        self._interaction_zone: list[float] | None = None
        logger.info(
            f"TaskExecutor initialized (rasim={rasim_url}, "
            f"max_iter={max_iterations}, delay={step_delay}s, "
            f"perception={perception_mode})"
        )

    @property
    def is_running(self) -> bool:
        """Whether a task is currently executing."""
        return self._running

    def cancel(self) -> bool:
        """Cancel the currently running task. Returns True if a task was running."""
        if self._cancel_event and self._running:
            logger.info("Cancellation requested")
            self._cancel_event.set()
            return True
        return False

    async def close(self) -> None:
        """Clean up HTTP client."""
        await self._client.aclose()

    async def _fetch_interaction_zone(self) -> list[float] | None:
        """Fetch and cache the interaction zone from RASim."""
        if self._interaction_zone is not None:
            return self._interaction_zone
        try:
            resp = await self._client.get(f"{self._rasim_url}/interaction-zone")
            resp.raise_for_status()
            self._interaction_zone = resp.json().get("interaction_zone")
            if self._interaction_zone:
                logger.info(f"Interaction zone: {self._interaction_zone}")
            return self._interaction_zone
        except Exception as e:
            logger.warning(f"Could not fetch interaction zone: {e}")
            return None

    async def _check_interaction_zone(self) -> dict:
        """Check if all props are within the interaction zone via RASim.

        Returns:
            Dict with 'ok' bool and 'violations' list.
        """
        try:
            resp = await self._client.get(
                f"{self._rasim_url}/interaction-zone/check"
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Interaction zone check failed: {e}")
            return {"ok": True, "violations": []}

    async def take_snapshot(
        self, width: int = 1024, height: int = 768
    ) -> dict[str, Any]:
        """Take an on-demand stereo perception snapshot.

        Captures images at the current arm position without moving to observe pose.
        Use for mid-task situation assessment.

        Returns:
            Snapshot data with images, camera params, scene state, and zone check.
        """
        try:
            resp = await self._client.get(
                f"{self._rasim_url}/snapshot",
                params={"width": width, "height": height},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info(
                f"Snapshot taken: {len(data.get('scene_state', {}).get('props', []))} props, "
                f"IZ ok={data.get('interaction_zone_check', {}).get('ok', '?')}"
            )
            return data
        except Exception as e:
            logger.error(f"Snapshot failed: {e}")
            return {"error": str(e)}

    def _check_fallen_props(self, state: dict[str, Any]) -> list[str]:
        """Check if any props have fallen off the table.

        A prop is considered fallen if its Z position is below the workspace
        surface height minus a tolerance (accounting for object size).
        Also checks X/Y bounds.

        Returns:
            List of fallen prop IDs (empty if all props are safe).
        """
        fallen = []
        workspace = state.get("workspace", {})
        surface_z = workspace.get("surface_height", 0.42)
        bounds = workspace.get("bounds", [0.1, 0.9, -0.4, 0.4])
        x_min, x_max, y_min, y_max = bounds

        # Generous margin for objects near edges
        margin = 0.15

        for prop in state.get("props", []):
            pos = prop.get("pos", [0, 0, 0])
            pid = prop.get("id", "unknown")

            # Fell below table (Z check with small tolerance for resting objects)
            if pos[2] < surface_z - 0.05:
                logger.warning(f"Prop {pid} fell below table: Z={pos[2]:.4f} (surface={surface_z})")
                fallen.append(pid)
                continue

            # Flew off the table laterally
            if (pos[0] < x_min - margin or pos[0] > x_max + margin or
                    pos[1] < y_min - margin or pos[1] > y_max + margin):
                logger.warning(f"Prop {pid} out of bounds: pos={pos}")
                fallen.append(pid)

        return fallen

    async def _get_scene_state(self) -> dict[str, Any]:
        """Fetch current scene state from RASim (or via stereo perception)."""
        if self._perception_mode == "camera" and self._perceiver is not None:
            logger.info("Perceiving scene via stereo cameras")
            return await self._perceiver.perceive(self._rasim_url)
        # Default: direct JSON scene state
        response = await self._client.get(f"{self._rasim_url}/scene-state")
        response.raise_for_status()
        return response.json()

    async def _get_camera_images(self) -> list[bytes] | None:
        """Get stereo camera images for visual verification (camera mode only)."""
        if self._perception_mode == "camera" and self._perceiver is not None:
            try:
                left, right = await self._perceiver.get_camera_images(self._rasim_url)
                return [left, right]
            except Exception as e:
                logger.warning(f"Failed to get camera images for verification: {e}")
        return None

    async def _execute_commands(self, commands: list[dict]) -> dict[str, Any]:
        """Send commands to RASim for execution."""
        response = await self._client.post(
            f"{self._rasim_url}/execute",
            json={"commands": commands},
            timeout=60.0,  # Arm movements can be slow
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
        self._cancel_event = asyncio.Event()
        self._running = True

        try:
            return await self._run_loop(task, execution_log)
        finally:
            self._running = False
            self._cancel_event = None

    async def _run_loop(
        self, task: str, execution_log: list[dict]
    ) -> dict[str, Any]:
        """Internal execution loop (separated for cancellation cleanup)."""
        # Fetch interaction zone at start
        await self._fetch_interaction_zone()

        # Pre-flight: verify all props are in the interaction zone before starting
        iz_check = await self._check_interaction_zone()
        if not iz_check.get("ok", True):
            violations = iz_check.get("violations", [])
            logger.error(f"Pre-flight IZ check failed: {violations}")
            return {
                "status": "interaction_zone_violation",
                "error": f"Cannot start: objects outside interaction zone",
                "violations": violations,
                "task": task,
                "iterations": 0,
                "log": execution_log,
            }

        for iteration in range(1, self._max_iterations + 1):
            logger.info(f"--- Iteration {iteration}/{self._max_iterations} ---")

            # Check for cancellation
            if self._cancel_event and self._cancel_event.is_set():
                logger.info("Task cancelled by user")
                return {
                    "status": "cancelled",
                    "task": task,
                    "iterations": iteration,
                    "log": execution_log,
                }

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
                    verify_images = await self._get_camera_images()
                    check = await self._verifier.verify(scene_state, task, images=verify_images)
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

                # Not actually done — inject failure context and retry
                failure_reason = check.get("reason", "unknown")
                logger.warning(f"Planner thinks done but verifier disagrees: {failure_reason}")
                execution_log.append({
                    "iteration": iteration,
                    "phase": "replanning",
                    "step_description": step_desc,
                    "commands": [],
                    "note": f"Previous attempt failed: {failure_reason}. Must re-attempt the task.",
                    "verification": check,
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

            # 4b. Safety check: abort if any prop fell off the table
            fallen = self._check_fallen_props(updated_state)
            if fallen:
                names = ", ".join(fallen)
                error_msg = f"Safety abort: prop(s) fell off the table: {names}"
                logger.error(error_msg)
                execution_log.append({
                    "iteration": iteration,
                    "phase": "safety_abort",
                    "fallen_props": fallen,
                })
                return {
                    "status": "safety_abort",
                    "error": error_msg,
                    "task": task,
                    "iterations": iteration,
                    "fallen_props": fallen,
                    "final_scene_state": updated_state,
                    "log": execution_log,
                }

            # 4c. Safety check: abort if any prop left the interaction zone
            iz_check = await self._check_interaction_zone()
            if not iz_check.get("ok", True):
                violations = iz_check.get("violations", [])
                violation_ids = [v["id"] for v in violations]
                error_msg = (
                    f"Interaction zone violation: {', '.join(violation_ids)} "
                    f"moved outside the safe interaction area"
                )
                logger.error(error_msg)
                execution_log.append({
                    "iteration": iteration,
                    "phase": "interaction_zone_violation",
                    "violations": violations,
                })
                return {
                    "status": "interaction_zone_violation",
                    "error": error_msg,
                    "task": task,
                    "iterations": iteration,
                    "violations": violations,
                    "final_scene_state": updated_state,
                    "log": execution_log,
                }

            # 5. Verify task completion
            try:
                verify_images = await self._get_camera_images()
                verification = await self._verifier.verify(updated_state, task, images=verify_images)
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

            # Check for cancellation before next iteration
            if self._cancel_event and self._cancel_event.is_set():
                logger.info("Task cancelled by user after execution step")
                return {
                    "status": "cancelled",
                    "task": task,
                    "iterations": iteration,
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
