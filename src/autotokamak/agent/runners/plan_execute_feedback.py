#!/usr/bin/env python3
"""Plan–execute runner with a global feedback loop.

Same interface as agent.plan_execute, but after the first plan+execute cycle the planner
is invoked again with the execution history; it can propose follow-up steps (e.g. fix
failures) or confirm completion. Repeats for up to feedback_rounds (config), then
optionally runs a validate_after review step.

Config (YAML) may include:
  problem: ...
  workspace: ...
  model: ...
  symlink: {...} or symlinks: [...]
  feedback_rounds: 2   # max plan+execute cycles (default 2)
  validate_after: true # run a post-execution review step (default false)
"""

import argparse

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

    feedback_rounds = max(1, int(getattr(cfg, "feedback_rounds", 2)))
    validate_after = getattr(cfg, "validate_after", False)

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

    execution_history: list[str] = []
    last_summary = "No previous step."

    for round_no in range(1, feedback_rounds + 1):
        if round_no == 1:
            planning_output = planner.invoke(problem)
        else:
            print("\n=== GLOBAL FEEDBACK: RE-PLAN (round {}) ===".format(round_no))
            replan_prompt = (
                f"Original problem:\n{problem}\n\n"
                f"Execution history so far:\n"
                + "\n---\n".join(execution_history)
                + "\n\n"
                f"Based on the above, suggest follow-up steps to fix failures or complete the task. "
                f"If nothing more is needed, return a plan with a single step: 'Confirm completion'."
            )
            planning_output = planner.invoke(replan_prompt)

        steps = planning_output["plan"].steps
        if not steps:
            print("No steps in plan; stopping.")
            break

        print("\n=== PLAN (round {}) ===".format(round_no))
        for i, s in enumerate(steps, 1):
            name = getattr(s, "name", f"Step {i}")
            desc = getattr(s, "description", str(s))
            print(f"  {i}. {name}\n     {desc}\n")

        print("\n=== EXECUTION (round {}) ===".format(round_no))
        last_summary = "No previous step."

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

        execution_history.append(last_summary)

        if (
            len(steps) == 1
            and "confirm completion" in last_summary.lower()
        ):
            print("\nPlanner confirmed completion; stopping feedback loop.")
            break

    if validate_after:
        print("\n=== VALIDATE (post-execution review) ===")
        validate_prompt = (
            f"Review the workspace and execution results for this task.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Execution summary:\n{last_summary}\n\n"
            f"Did the task succeed? If not, what failed or is missing? Be brief."
        )
        result = executor.invoke(
            {
                "messages": [HumanMessage(content=validate_prompt)],
                "workspace": workspace,
                "symlinkdir": symlinkdict,
            }
        )
        print(result["messages"][-1].text)

    print("\n=== FINAL ===")
    print(last_summary)
    print(f"\nWorkspace: {workspace_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plan–execute with global feedback loop (re-plan after execution).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (problem, workspace, feedback_rounds, validate_after, etc.).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model string for init_chat_model (e.g. openai:gpt-5-mini).",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override workspace directory.",
    )
    args = parser.parse_args()
    main(args.config, args.model, args.workspace)
