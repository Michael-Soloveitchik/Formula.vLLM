#!/usr/bin/env python3
"""Solver harness using the Claude Agent SDK (no IBM LiteLLM needed).

Given a task work dir containing ./vllm (the base_sha tree) and problem.md, the
agent edits ONLY the listed source files to implement the optimization, then
exits. The runner (run.sh) verifies the result against the hidden test_patch.

Usage:
    python3 claude_agent_harness.py <work_dir> <problem_md> <src_files_csv> [model]

Requires: pip install claude-agent-sdk ; env ANTHROPIC_API_KEY.
Model defaults to a fast Sonnet; override via arg or config.yaml.
"""
import asyncio
import os
import sys


async def solve(work_dir, problem_md, src_files, model):
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

    problem = open(problem_md).read()
    prompt = f"""{problem}

You are optimizing vLLM. The package is at ./vllm in the current directory.
Edit ONLY these source files to implement the behavior + performance objective:
{src_files}

Do NOT read or edit any test files. Keep public behavior correct. Make the
smallest change that achieves the objective. When done, stop."""

    opts = ClaudeAgentOptions(
        cwd=work_dir,
        model=model,
        allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
    )
    async with ClaudeSDKClient(options=opts) as client:
        await client.query(prompt)
        async for _ in client.receive_response():
            pass


def main():
    if len(sys.argv) < 4:
        print("usage: claude_agent_harness.py <work_dir> <problem_md> "
              "<src_files_csv> [model]", file=sys.stderr)
        sys.exit(2)
    work_dir, problem_md, src_csv = sys.argv[1:4]
    model = sys.argv[4] if len(sys.argv) > 4 else os.environ.get(
        "CLAUDE_MODEL", "claude-sonnet-4-6")
    asyncio.run(solve(work_dir, problem_md, src_csv, model))


if __name__ == "__main__":
    main()
