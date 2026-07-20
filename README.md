# LLM Browser Agent

An autonomous web-browsing agent driven by a **local Llama 3.1 8B** model (served via **Ollama**) and [**browser-use**](https://github.com/browser-use/browser-use). Given a natural-language goal, the agent navigates a real Chromium browser, reads the page, decides its next action, and repeats until the task is done — no hardcoded scraping rules.

Built and run on a RunPod GPU pod (NVIDIA RTX 6000 Ada, 48 GB VRAM).

---

## Example

Given the task *"Navigate to the OpenAI Wikipedia page, read it, and describe what OpenAI does in one sentence,"* the agent autonomously navigated, searched, self-corrected its approach, and returned:

> *"The mission statement of OpenAI is to ensure that artificial general intelligence benefits all of humanity."*

...in 7 steps, including recovering from a failed search strategy on its own (details in [Results](#results)).

---

## How it works

A language model can only read and write text — it can't click a button or read a live web page on its own. browser-use gives it eyes and hands, and runs it in a loop.

**Two pieces:**

- **browser-use** — a Python library that drives a real Chromium browser via Playwright. It turns the current page into a text/accessibility description the model can understand, and executes the actions the model chooses (click, scroll, type, extract, done).
- **Ollama serving Llama 3.1 8B** — runs the model locally on the GPU. The model receives the page description plus the goal and decides the next action.

**The loop (observe → decide → act):**

1. browser-use loads the page and builds a text description of it.
2. That description + the goal go to the model.
3. The model replies with the next action (e.g. *extract the H1*, *scroll*, *done*).
4. browser-use executes that action in the browser.
5. The page changes; browser-use reads the new state.
6. Repeat until the model calls `done` or hits the step cap.

This is the standard agentic tool-use loop, specialized for web browsing.

---

## Tech stack

| Component | Value |
|---|---|
| Model | `llama3.1:8b` (text-only) |
| Model server | Ollama |
| Agent framework | browser-use 0.13.1 |
| Browser automation | Playwright + Chromium |
| Language | Python 3.12 |
| Hardware | RunPod GPU pod — NVIDIA RTX 6000 Ada, 48 GB VRAM (CUDA 12.4) |

### Why self-hosted

Ollama serves the model locally on the pod's GPU and exposes it on `localhost:11434`, which browser-use connects to. Everything runs self-hosted — no external API dependency and no per-call cost.

---

## Engineering challenges & fixes

Getting a text-only 8B model to reliably drive a browser meant solving three concrete problems:

- **Playwright CLI not on PATH** — the `playwright` command wasn't available after install, so I invoked it via `python3 -m playwright install chromium`.
- **`multimodal not supported` 400 error** — browser-use defaults to *vision mode* and sends page screenshots alongside the text. Llama 3.1 8B is text-only and rejects image input with a 400. Setting **`use_vision=False`** sends only the text/accessibility tree, which the model handles — and it runs faster.
- **Runaway loop (112 steps)** — an early run looped indefinitely because the small model couldn't recognize when the task was complete. Fixed with **`max_steps=10`** as a hard cap, plus an explicit, ordered prompt with a clear stop condition.

---

## Setup

Run these on the machine serving Ollama (here, a RunPod pod terminal).

**1. Install and start Ollama, pull the model:**

```bash
curl -fsSL https://ollama.com/install.sh | sh

# start the server in the background
ollama serve > ollama.log 2>&1 &

# download the 8B model (~4.7 GB)
ollama pull llama3.1:8b
```

> On RunPod, point `OLLAMA_MODELS` at the persistent volume so the model survives a pod restart:
> `export OLLAMA_MODELS=/workspace/ollama_models`

**2. Install browser-use + Playwright + Chromium:**

```bash
pip install -r requirements.txt
python3 -m playwright install chromium
python3 -m playwright install-deps chromium
```

---

## Usage

Pass the agent a natural-language task:

```bash
# uses the default OpenAI-Wikipedia task
python3 browser_agent.py

# or provide your own
python3 browser_agent.py "Go to example.com and report the main heading"
```

Optional flags: `--model`, `--host`, and `--max-steps` (defaults: `llama3.1:8b`, `http://localhost:11434`, `10`). Host and model can also be set via the `OLLAMA_HOST` and `OLLAMA_MODEL` environment variables.

The agent connects to Ollama, launches Chromium, and prints the final result. Because it talks to a locally served model, it must run on the same machine that's serving Ollama.

---

## Results

The agent was evaluated on three tasks of increasing difficulty — showing both a working agent and the characteristic limits of a small (8B) local model as the reasoning brain.

**Run 1 — Controlled baseline (`example.com`): report the main H1.**
Clean success in **2 steps**. Confirmed the whole chain works: Ollama connection, Chromium launch, page reading, the loop, and stopping properly.

**Run 2 — Two-part task (OpenAI Wikipedia: what it does *and* year founded).**
Success in **2 steps**, but the model answered only half. Its step-1 memory registered *"OpenAI, a company focused on artificial intelligence research,"* but it only extracted the founding year, then called `done`. It had the information and still dropped half the task — small models don't reliably hold a two-part instruction all the way through.

**Run 3 — Single fact (OpenAI Wikipedia: describe what OpenAI does).**
Success in **7 steps**, after inefficient wandering and a self-correction. It first searched the page for literal phrases (*"what OpenAI does,"* *"mission statement"*) → 0 matches (one step hit a 180s timeout). It then switched to the semantic `extract` tool, found the mission statement, and finished. A larger model would likely go straight to extraction.

---

## Findings & limitations

**What works well**
- The full pipeline works end to end: a local 8B model + browser-use + real Chromium completing autonomous web tasks.
- On simple, well-scoped tasks the agent is fast and reliable (Run 1).
- With an explicit prompt and a step cap, it completes realistic extraction tasks (Run 3).

**Limitations of an 8B local model as the agent brain**
- **Multi-part instructions are unreliable** — in Run 2 it completed one sub-goal and stopped.
- **It takes inefficient reasoning paths** — in Run 3 it tried literal text search before switching to semantic extraction.
- **It can loop indefinitely** without a `max_steps` cap when it can't tell whether the task is finished (an early run hit 112 steps).

**Mitigations applied**
- `use_vision=False` to match the text-only model.
- Explicit, ordered task prompts with a clear stop condition.
- `max_steps=10` as a hard safety cap.

**Possible improvements (not implemented)**
- Use a **vision-capable** model (e.g. `llama3.2-vision`) to handle visually complex pages.
- Use a **larger or more agent-tuned** model to improve multi-step reliability and reduce wandering (the 48 GB GPU has room).
- Add a `fallback_llm` so a single provider error doesn't stall the run.
