# LLM Browser Agent

An autonomous web browsing agent that runs on a local Llama 3.1 8B model (served with Ollama) and browser-use. You give it a goal in plain English, and it opens a real Chromium browser, reads the page, decides what to do next, and repeats until the task is done. There are no hardcoded scraping rules.

I built and ran this on a RunPod GPU pod (NVIDIA RTX 6000 Ada, 48 GB VRAM).

## Example

I gave it the task "Navigate to the OpenAI Wikipedia page, read it, and describe what OpenAI does in one sentence." It navigated, searched, fixed its own approach after a search that returned nothing, and came back with:

"The mission statement of OpenAI is to ensure that artificial general intelligence benefits all of humanity."

That took 7 steps, including recovering from a bad search strategy on its own. There's more on that run below.

## How it works

A language model can only read and write text. It can't click a button or read a live web page on its own. To make it act on the web, you pair it with a tool that gives it a way to see the page and act on it, and you run that in a loop. browser-use is that tool.

There are two pieces.

browser-use is a Python library that drives a real Chromium browser through Playwright. It reads the current page, turns it into a text description the model can understand, and runs the actions the model picks (click, scroll, type, extract, done).

Ollama runs Llama 3.1 8B locally on the machine. The model gets the page description and the goal, and decides the next action.

The loop is observe, decide, act:

1. browser-use loads the page and builds a text description of it.
2. That description and the goal go to the model.
3. The model replies with the next action, like extract the H1, scroll, or done.
4. browser-use runs that action in the browser.
5. The page changes, and browser-use reads the new state.
6. This repeats until the model calls done or hits the step cap.

This is the basic loop behind any agent. browser-use is the version of it built for web browsing.

## Tech stack

| Component | Value |
|---|---|
| Model | llama3.1:8b (text-only) |
| Model server | Ollama |
| Agent framework | browser-use 0.13.1 |
| Browser automation | Playwright + Chromium |
| Language | Python 3.12 |
| Hardware | RunPod GPU pod, NVIDIA RTX 6000 Ada, 48 GB VRAM, CUDA 12.4 |

Everything is self-hosted. Ollama runs the model on the machine's GPU and exposes it on localhost:11434, and browser-use connects to that. There's no external API to call and no per-call cost.

## Problems I ran into

Getting a text-only 8B model to drive a browser took solving three things.

The Playwright CLI wasn't on my PATH after install, so I ran it with `python3 -m playwright install chromium` instead.

The agent threw a "multimodal not supported" 400 error. browser-use sends page screenshots to the model by default, and Llama 3.1 8B is text-only, so it rejects images. I set `use_vision=False`, which sends only the text version of the page. That fixed it, and it runs faster too.

An early run looped to 112 steps because the model couldn't tell when it was done. I added `max_steps=10` as a hard cap and wrote a more explicit prompt with a clear stop instruction.

## Setup

Run these on the machine serving Ollama.

Install and start Ollama, then pull the model:

```bash
curl -fsSL https://ollama.com/install.sh | sh

# start the server in the background
ollama serve > ollama.log 2>&1 &

# download the model (~4.7 GB)
ollama pull llama3.1:8b
```

On RunPod, point `OLLAMA_MODELS` at the persistent volume so the model survives a restart:

```bash
export OLLAMA_MODELS=/workspace/ollama_models
```

Install browser-use, Playwright, and Chromium:

```bash
pip install -r requirements.txt
python3 -m playwright install chromium
python3 -m playwright install-deps chromium
```

## Usage

Give the agent a task in plain English:

```bash
# runs the default task
python3 browser_agent.py

# or your own
python3 browser_agent.py "Go to https://example.com and report the main H1 heading"
```

You can also set `--model`, `--host`, and `--max-steps`. The defaults are llama3.1:8b, http://localhost:11434, and 10. The host and model can also come from the `OLLAMA_HOST` and `OLLAMA_MODEL` environment variables.

It connects to Ollama, opens Chromium, and prints the result. Because it talks to a local model, it has to run on the same machine that's serving Ollama.

## Results

I tested it on three tasks that get harder as they go. Together they show the agent working and where a small (8B) local model starts to struggle.

Run 1, controlled baseline (example.com), report the main H1. Clean success in 2 steps. This confirmed the whole chain works: the Ollama connection, Chromium launching, reading the page, running the loop, and stopping properly.

Run 2, two-part task (OpenAI Wikipedia, what it does and the year founded). Success in 2 steps, but it only answered half. Its memory from step 1 had noted "OpenAI, a company focused on artificial intelligence research," but it only pulled the founding year and then called done. It had the information and still dropped half the task. A small model doesn't reliably hold a two-part instruction all the way through.

Run 3, single fact (OpenAI Wikipedia, describe what it does). Success in 7 steps after some wandering and a self-correction. It first searched the page for the literal phrases "what OpenAI does" and "mission statement," which found nothing because those exact words aren't on the page (one step hit a 180 second timeout). Then it switched to the extract tool, which works on meaning instead of exact text, and found the answer. A bigger model would probably go straight to extraction.

## What I learned

What works. The full pipeline runs end to end. A local 8B model, browser-use, and a real Chromium browser can complete web tasks on their own. On simple, well-scoped tasks it's fast and reliable (Run 1). With an explicit prompt and a step cap, it handles realistic extraction tasks too (Run 3).

Where an 8B model falls short as the brain. It's unreliable on multi-part instructions, like Run 2 where it finished one part and stopped. It takes inefficient paths, like the literal search before the semantic extract in Run 3. And without the step cap it can loop forever when it can't tell it's done, which is how the early run hit 112 steps.

What I did about it. I set `use_vision=False` to match the text-only model, wrote explicit and ordered prompts with a clear stop condition, and capped the run at 10 steps.

If I kept going. I'd use a vision model like llama3.2-vision for pages that need to be seen and not just read, use a bigger or more agent-tuned model to cut down the wandering and handle multi-step tasks better (the 48 GB card has room), and add a fallback model so one provider error doesn't kill the run.
