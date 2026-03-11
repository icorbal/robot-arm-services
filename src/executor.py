"""Plan-execute-verify loop - orchestrates the full task execution pipeline."""

import asyncio
import logging
from typing import Any

import httpx

from .perception import Perceiver
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
        perceiver: Perceiver | None = None,
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
        """Take an on-demand perception snapshot.

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

    async def _get_scene_state(
        self, refined: bool = False,
        target_objects: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Fetch current scene state from RASim (or via camera perception).

        Args:
            refined: If True, use two-phase perception (Phase 1 scan +
                Phase 2 targeted close-ups). If False (default), use only
                Phase 1 scan for a quick overview.
            target_objects: When refined=True, optionally limit Phase 2 to
                specific objects. Each dict has 'id' and 'description'.
        """
        if self._perception_mode == "camera" and self._perceiver is not None:
            if refined:
                logger.info("Perceiving scene via two-phase camera pipeline (refined)")
                return await self._perceiver.perceive_two_phase(
                    self._rasim_url,
                    target_objects=target_objects,
                )
            else:
                logger.info("Perceiving scene via single-scan camera pipeline")
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

    async def _verify_grab(
        self,
        target_id: str,
        target_description: str,
        pre_grab_pos: list[float],
    ) -> dict[str, Any]:
        """Verify that an object was successfully grabbed after grip_close.

        Observes the scene and checks whether the target object is still at
        its previous position (grab missed) or has moved with the gripper
        (grab succeeded).

        Args:
            target_id: ID of the object we tried to grab.
            target_description: e.g. "red box" for logging.
            pre_grab_pos: Object position before the grab attempt.

        Returns:
            Dict with 'grabbed' (bool) and 'details'.
        """
        if self._perception_mode != "camera" or self._perceiver is None:
            # Without camera perception, assume grab succeeded
            return {"grabbed": True, "details": "no camera verification available"}

        logger.info(f"Verifying grab of '{target_id}' ({target_description})")

        # Do a targeted observation focused on where the object was
        refined = await self._perceiver.perceive_targeted(
            self._rasim_url,
            pre_grab_pos,
            target_description,
        )

        if refined is None:
            # Object not found at previous location — likely grabbed
            logger.info(f"Grab verification: '{target_id}' not found at "
                        f"previous position — likely grabbed ✅")
            return {"grabbed": True, "details": "object not visible at previous position"}

        import numpy as np
        dist = np.linalg.norm(
            np.array(refined["pos"]) - np.array(pre_grab_pos)
        )

        if dist < 0.03:
            # Object still at same position — grab missed
            logger.warning(
                f"Grab verification FAILED: '{target_id}' still at "
                f"[{refined['pos'][0]:.3f}, {refined['pos'][1]:.3f}, "
                f"{refined['pos'][2]:.3f}] (moved {dist*100:.1f}cm) ❌"
            )
            return {
                "grabbed": False,
                "details": f"object still at original position (moved only {dist*100:.1f}cm)",
                "refined_pos": refined["pos"],
            }
        else:
            # Object moved significantly — grab succeeded
            logger.info(
                f"Grab verification: '{target_id}' moved {dist*100:.1f}cm "
                f"from previous position — grab confirmed ✅"
            )
            return {"grabbed": True, "details": f"object moved {dist*100:.1f}cm"}

    async def _check_grab_in_commands(
        self,
        commands: list[dict],
        scene_state: dict[str, Any],
        execution_log: list[dict],
        iteration: int,
    ) -> bool | None:
        """Check if a grip_close in the command sequence actually grabbed an object.

        Looks for a grip_close command preceded by a move_to, finds the
        closest object to that move_to position, and verifies the grab.

        Returns:
            True if grab confirmed, False if grab failed, None if no
            grip_close in commands.
        """
        # Find grip_close and the preceding move_to
        grip_idx = None
        for i, cmd in enumerate(commands):
            if cmd.get("type") == "grip_close":
                grip_idx = i
                break

        if grip_idx is None:
            return None  # No grab in this step

        # Find the last move_to before grip_close to determine target position
        grab_pos = None
        for i in range(grip_idx - 1, -1, -1):
            if commands[i].get("type") == "move_to":
                grab_pos = commands[i].get("position")
                break

        if grab_pos is None:
            return None  # Can't determine grab location

        # Find closest object to grab position
        import numpy as np
        props = scene_state.get("props", [])
        best_id = None
        best_dist = float("inf")
        best_pos = None
        best_desc = None
        for p in props:
            d = np.linalg.norm(np.array(p["pos"][:2]) - np.array(grab_pos[:2]))
            if d < best_dist:
                best_dist = d
                best_id = p["id"]
                best_pos = p["pos"]
                best_desc = f"{p.get('color', '')} {p.get('type', 'object')}".strip()

        if best_id is None or best_dist > 0.10:
            logger.warning(f"No object near grab position {grab_pos}")
            return None

        logger.info(f"Grab target: '{best_id}' ({best_desc}) at "
                    f"[{best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}]")

        # Verify the grab
        result = await self._verify_grab(best_id, best_desc, best_pos)

        if not result["grabbed"]:
            # Inject failure context for replanning
            execution_log.append({
                "iteration": iteration,
                "phase": "grab_failed",
                "target_id": best_id,
                "target_description": best_desc,
                "details": result["details"],
                "refined_pos": result.get("refined_pos"),
                "note": (
                    f"Grab of '{best_id}' failed — object still at original "
                    f"position. Need to re-perceive and retry."
                ),
            })
            return False

        return True

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

        use_refined_next = False  # Escalate to two-phase on failure

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

            # 1. Get scene state (use refined if previous iteration failed)
            try:
                scene_state = await self._get_scene_state(refined=use_refined_next)
                use_refined_next = False  # Reset after use
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

                # Not actually done — inject failure context and retry with refined perception
                failure_reason = check.get("reason", "unknown")
                logger.warning(f"Planner thinks done but verifier disagrees: {failure_reason}")
                use_refined_next = True
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

            # 3b. Grab verification: after grip_close, check if we
            # actually got the object.  If not, re-perceive with
            # targeted close-up and inject failure context for replanning.
            grab_failed = False
            if self._perception_mode == "camera" and self._perceiver is not None:
                grab_verified = await self._check_grab_in_commands(
                    commands, scene_state, execution_log, iteration
                )
                if grab_verified is False:
                    grab_failed = True

            # 4. Get updated scene state
            try:
                # Use refined perception if grab failed (need accurate re-perception)
                updated_state = await self._get_scene_state(refined=grab_failed)
            except httpx.HTTPError as e:
                logger.error(f"Failed to get updated scene state: {e}")
                updated_state = exec_result.get("scene_state", scene_state)

            if grab_failed:
                # Skip verification — force replanning with refined positions
                continue

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

            # Task not complete yet — escalate to refined perception for next attempt
            use_refined_next = True

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
