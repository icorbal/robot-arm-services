"""Integration tests for Robot Arm Services with mocked RASim and LLM."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.executor import TaskExecutor
from src.planner import TaskPlanner
from src.verifier import TaskVerifier


# --- Fixtures ---

MOCK_SCENE_STATE = {
    "arm": {
        "gripper_pos": [0.3, 0.0, 0.7],
        "gripper_rot": [0.0, 0.0, 0.0],
        "grip_open": True,
        "joint_positions": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    },
    "props": [
        {
            "id": "blue_box",
            "type": "box",
            "color": "blue",
            "pos": [0.5, -0.15, 0.45],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.03, 0.03, 0.03],
        },
        {
            "id": "red_box",
            "type": "box",
            "color": "red",
            "pos": [0.5, 0.15, 0.45],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.03, 0.03, 0.03],
        },
    ],
    "workspace": {
        "bounds": [0.1, 0.9, -0.4, 0.4],
        "surface_height": 0.42,
    },
}

MOCK_SCENE_STATE_PICKED_UP = {
    **MOCK_SCENE_STATE,
    "arm": {
        **MOCK_SCENE_STATE["arm"],
        "gripper_pos": [0.5, -0.15, 0.6],
        "grip_open": False,
    },
    "props": [
        {
            "id": "blue_box",
            "type": "box",
            "color": "blue",
            "pos": [0.5, -0.15, 0.6],
            "rot": [0.0, 0.0, 0.0],
            "size": [0.03, 0.03, 0.03],
        },
        MOCK_SCENE_STATE["props"][1],
    ],
}


def create_mock_llm():
    """Create a mock LLM adapter."""
    mock = AsyncMock()
    mock.close = AsyncMock()
    return mock


# --- Planner Tests ---

@pytest.mark.asyncio
async def test_planner_pick_up_step():
    """Test that the planner generates a pick-up plan."""
    mock_llm = create_mock_llm()
    mock_llm.generate.return_value = {
        "step_description": "Approach blue box from above",
        "commands": [
            {"type": "grip_open"},
            {"type": "move_to", "position": [0.5, -0.15, 0.55]},
        ],
    }

    planner = TaskPlanner(mock_llm)
    result = await planner.plan_next_step(MOCK_SCENE_STATE, "pick up the blue box")

    assert "commands" in result
    assert len(result["commands"]) == 2
    assert result["commands"][0]["type"] == "grip_open"
    assert result["commands"][1]["type"] == "move_to"
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_planner_empty_commands_when_done():
    """Test planner returns empty commands when task is complete."""
    mock_llm = create_mock_llm()
    mock_llm.generate.return_value = {
        "step_description": "Task already complete",
        "commands": [],
    }

    planner = TaskPlanner(mock_llm)
    result = await planner.plan_next_step(MOCK_SCENE_STATE_PICKED_UP, "pick up the blue box")

    assert result["commands"] == []


# --- Verifier Tests ---

@pytest.mark.asyncio
async def test_verifier_not_completed():
    """Test verifier detects incomplete task."""
    mock_llm = create_mock_llm()
    mock_llm.generate.return_value = {
        "completed": False,
        "reason": "Blue box is still on the table surface",
        "confidence": 0.9,
    }

    verifier = TaskVerifier(mock_llm)
    result = await verifier.verify(MOCK_SCENE_STATE, "pick up the blue box")

    assert result["completed"] is False
    assert "reason" in result
    assert result["confidence"] == 0.9


@pytest.mark.asyncio
async def test_verifier_completed():
    """Test verifier detects completed task."""
    mock_llm = create_mock_llm()
    mock_llm.generate.return_value = {
        "completed": True,
        "reason": "Blue box is lifted above the table surface",
        "confidence": 0.95,
    }

    verifier = TaskVerifier(mock_llm)
    result = await verifier.verify(MOCK_SCENE_STATE_PICKED_UP, "pick up the blue box")

    assert result["completed"] is True


# --- Executor Tests ---

@pytest.mark.asyncio
async def test_executor_pick_up_loop():
    """Test the full plan-execute-verify loop for picking up a box."""
    mock_llm = create_mock_llm()

    # Sequence: plan approach -> plan grasp -> plan lift -> verify done
    mock_llm.generate.side_effect = [
        # Step 1: Plan - approach
        {
            "step_description": "Open gripper and approach blue box",
            "commands": [
                {"type": "grip_open"},
                {"type": "move_to", "position": [0.5, -0.15, 0.55]},
            ],
        },
        # Step 1: Verify - not done
        {"completed": False, "reason": "Box not grasped yet", "confidence": 0.9},
        # Step 2: Plan - lower and grasp
        {
            "step_description": "Lower to blue box and grasp",
            "commands": [
                {"type": "move_to", "position": [0.5, -0.15, 0.45]},
                {"type": "grip_close"},
            ],
        },
        # Step 2: Verify - not done
        {"completed": False, "reason": "Box grasped but not lifted", "confidence": 0.8},
        # Step 3: Plan - lift
        {
            "step_description": "Lift the blue box",
            "commands": [
                {"type": "move_to", "position": [0.5, -0.15, 0.6]},
            ],
        },
        # Step 3: Verify - done!
        {"completed": True, "reason": "Blue box is lifted above table", "confidence": 0.95},
    ]

    planner = TaskPlanner(mock_llm)
    verifier = TaskVerifier(mock_llm)

    executor = TaskExecutor(
        planner=planner,
        verifier=verifier,
        rasim_url="http://mock:8100",
        max_iterations=10,
        step_delay=0.0,  # No delay in tests
    )

    # Mock HTTP client
    mock_response_state = MagicMock()
    mock_response_state.status_code = 200
    mock_response_state.json.return_value = MOCK_SCENE_STATE
    mock_response_state.raise_for_status = MagicMock()

    mock_response_exec = MagicMock()
    mock_response_exec.status_code = 200
    mock_response_exec.json.return_value = {
        "commands_executed": 2,
        "results": [{"command": "move_to", "success": True}],
        "scene_state": MOCK_SCENE_STATE,
    }
    mock_response_exec.raise_for_status = MagicMock()

    call_count = {"get": 0}

    async def mock_get(url):
        call_count["get"] += 1
        # After step 3, return picked up state
        if call_count["get"] >= 5:
            picked = MagicMock()
            picked.status_code = 200
            picked.json.return_value = MOCK_SCENE_STATE_PICKED_UP
            picked.raise_for_status = MagicMock()
            return picked
        return mock_response_state

    async def mock_post(url, json=None):
        return mock_response_exec

    executor._client.get = mock_get
    executor._client.post = mock_post

    result = await executor.execute_task("pick up the blue box")

    assert result["status"] == "completed"
    assert result["iterations"] == 3
    assert len(result["log"]) == 3
    assert result["verification"]["completed"] is True


@pytest.mark.asyncio
async def test_executor_max_iterations():
    """Test executor stops at max iterations."""
    mock_llm = create_mock_llm()

    # Always return commands and never verify as complete
    mock_llm.generate.side_effect = lambda *args, **kwargs: (
        {"step_description": "Try again", "commands": [{"type": "grip_open"}]}
        if "Plan" in args[1]
        else {"completed": False, "reason": "Not done", "confidence": 0.5}
    )

    planner = TaskPlanner(mock_llm)
    verifier = TaskVerifier(mock_llm)

    executor = TaskExecutor(
        planner=planner,
        verifier=verifier,
        rasim_url="http://mock:8100",
        max_iterations=3,
        step_delay=0.0,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_SCENE_STATE
    mock_resp.raise_for_status = MagicMock()

    mock_exec_resp = MagicMock()
    mock_exec_resp.status_code = 200
    mock_exec_resp.json.return_value = {
        "commands_executed": 1,
        "results": [{"command": "grip_open", "success": True}],
        "scene_state": MOCK_SCENE_STATE,
    }
    mock_exec_resp.raise_for_status = MagicMock()

    async def mock_get(url):
        return mock_resp

    async def mock_post(url, json=None):
        return mock_exec_resp

    executor._client.get = mock_get
    executor._client.post = mock_post

    result = await executor.execute_task("impossible task")

    assert result["status"] == "max_iterations_reached"
    assert result["iterations"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
