import argparse
from pathlib import Path

from dotenv import load_dotenv

from agent.runners.config import (
    REPO_ROOT,
    load_config,
    materialize_symlinks,
    resolve_workspace,
)

load_dotenv(REPO_ROOT / ".env")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from ursa.agents import ExecutionAgent, PlanningAgent


def main(config_path: str, cli_model: str | None, workspace_override: str | None):
    cfg = load_config(config_path)

    problem = getattr(cfg, "problem", None)
    if not problem:
        raise ValueError("config.yaml must contain a top-level 'problem:' string")

    model_name = (
        cli_model
        or getattr(cfg, "model", None)
        or "openai:gpt-5-mini"
    )

    print(f"\nUsing model: {model_name}")

    workspace_path = resolve_workspace(
        workspace_override
        or getattr(cfg, "workspace", None)
        or "mini_workspace"
    )
    workspace_path.mkdir(parents=True, exist_ok=True)
    workspace = str(workspace_path)

    # URSA only supports a single symlinkdir dict; we materialize the YAML's
    # `symlinks:` list ourselves and pass None to URSA to skip its broken path.
    symlink_entries = getattr(cfg, "symlinks", None) or getattr(cfg, "symlink", None)
    if isinstance(symlink_entries, dict):
        symlink_entries = [symlink_entries]
    materialize_symlinks(workspace_path, symlink_entries)

    planner_llm = init_chat_model(model=model_name)
    executor_llm = init_chat_model(model=model_name)

    planner = PlanningAgent(
        llm=planner_llm,
        thread_id="demo_planner",
        workspace=workspace,
    )

    executor = ExecutionAgent(
        llm=executor_llm,
        thread_id="demo_executor",
        workspace=workspace,
    )

    planning_output = planner.invoke(problem)
    steps = planning_output["plan"].steps

    print("\n=== PLAN ===")
    for i, s in enumerate(steps, 1):
        name = getattr(s, "name", f"Step {i}")
        desc = getattr(s, "description", str(s))
        print(f"{i}. {name}\n   {desc}\n")

    last_summary = "No previous step."
    print("\n=== EXECUTION ===")

    for i, step in enumerate(steps, 1):
        step_text = (
            f"{getattr(step, 'name', f'Step {i}')}\n"
            f"{getattr(step, 'description', str(step))}"
        )

        prompt = (
            f"You are executing a multi-step plan.\n\n"
            f"Overall problem:\n{problem}\n\n"
            f"Previous-step summary:\n{last_summary}\n\n"
            f"Current step:\n{step_text}\n\n"
            f"Execute this step fully. Use tools if helpful. "
            f"If you write code, save it in the workspace.\n"
        )

        result = executor.invoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "workspace": workspace,
                "symlinkdir": None,
            }
        )

        last_summary = result["messages"][-1].text
        print(f"\n--- Step {i} result ---\n{last_summary}")

    print("\n=== FINAL ===")
    print(last_summary)
    print(f"\nWorkspace: {workspace_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="Model string for init_chat_model (e.g. openai:gpt-5-mini)",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override workspace directory (optional)",
    )
    args = parser.parse_args()

    main(args.config, args.model, args.workspace)
