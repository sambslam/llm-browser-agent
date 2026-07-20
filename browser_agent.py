"""
LLM Browser Agent — an autonomous web-browsing agent driven by a local
Llama 3.1 8B model (served via Ollama) and browser-use.

Given a natural-language task, the agent navigates a real Chromium browser,
reads each page as text, decides the next action, and repeats until the task
is done (observe -> decide -> act).

Usage:
    python browser_agent.py                       # runs the default task
    python browser_agent.py "Go to example.com and report the main heading"
    python browser_agent.py --task "..." --max-steps 10

Must run on the machine serving Ollama (defaults to localhost:11434).
"""

import argparse
import asyncio
import os

from browser_use import Agent
from browser_use.llm import ChatOllama

DEFAULT_TASK = (
    "Navigate to https://en.wikipedia.org/wiki/OpenAI. Read the page. "
    "In one sentence, describe what OpenAI does. Once you have this, "
    "immediately mark the task done and stop."
)


async def run_agent(task: str, model: str, host: str, max_steps: int):
    # Connect browser-use to the locally served Ollama model
    llm = ChatOllama(model=model, host=host)

    agent = Agent(
        task=task,
        llm=llm,
        use_vision=False,  # text-only model: do not send screenshots
    )

    result = await agent.run(max_steps=max_steps)  # hard cap to prevent runaway loops
    print("\n\n=== AGENT RESULT ===")
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous browser agent (browser-use + a local Ollama model)."
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=DEFAULT_TASK,
        help="Natural-language task for the agent (defaults to the OpenAI-Wikipedia example).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        help="Ollama model name (default: llama3.1:8b).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Hard cap on agent steps to prevent runaway loops (default: 10).",
    )
    args = parser.parse_args()

    asyncio.run(run_agent(args.task, args.model, args.host, args.max_steps))


if __name__ == "__main__":
    main()
