#!/usr/bin/env python3
"""Entry point for Robot Arm Services (RAServ)."""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn
import yaml

from src.api import app, set_executor
from src.executor import TaskExecutor
from src.llm_adapter import create_llm_adapter
from src.planner import TaskPlanner
from src.verifier import TaskVerifier


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_settings(path: str | Path) -> dict:
    """Load settings from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Robot Arm Services (RAServ)")
    parser.add_argument(
        "--port", type=int, default=8200,
        help="API server port (default: 8200)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="API server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/settings.yaml",
        help="Path to settings YAML"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting Robot Arm Services (RAServ)")

    # Load settings
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    settings = load_settings(config_path)
    logger.info(f"Loaded settings from {config_path}")

    # Create LLM adapter
    llm_config = settings.get("llm", {})
    try:
        llm = create_llm_adapter(
            provider=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o"),
            api_key_env=llm_config.get("api_key_env", "OPENAI_API_KEY"),
        )
    except ValueError as e:
        logger.error(f"Failed to create LLM adapter: {e}")
        sys.exit(1)

    # Create planner and verifier
    planner = TaskPlanner(llm)
    verifier = TaskVerifier(llm)

    # Create executor
    exec_config = settings.get("executor", {})
    rasim_config = settings.get("rasim", {})
    executor = TaskExecutor(
        planner=planner,
        verifier=verifier,
        rasim_url=rasim_config.get("url", "http://localhost:8100"),
        max_iterations=exec_config.get("max_iterations", 10),
        step_delay=exec_config.get("step_delay", 0.5),
    )
    set_executor(executor)

    # Start API server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
