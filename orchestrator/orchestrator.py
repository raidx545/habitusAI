"""
orchestrator.py — Claude-replacement Orchestrator using Groq API
Parses user intent, dispatches 4 agents in parallel, then hands off to aggregator.
"""

import asyncio
import json
from groq import Groq
from contracts import AgentInput

# ── Groq config ───────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"

def _get_client() -> Groq:
    """Lazy init — only fails if GROQ_API_KEY missing when actually called."""
    return Groq()  # reads GROQ_API_KEY from env

SYSTEM_PROMPT = """You are an interior design orchestrator AI.
Given a user's room photo URL and their design request, your job is to:
1. Parse and clarify the user's design intent into structured form.
2. Identify the dominant style category they are aiming for.
3. Extract any budget or constraint information.
4. Return a clean JSON object — no prose, no markdown, just raw JSON.

Output JSON shape:
{
  "parsed_intent": "<one clear sentence summarising what the user wants>",
  "style_hint": "<one of: minimalist, maximalist, scandinavian, japandi, industrial, bohemian, coastal, contemporary>",
  "budget_inr": <integer or null>,
  "priority_rooms": ["<item>"],
  "extra_constraints": "<any Vastu, accessibility, or special notes>"
}"""


def parse_intent(agent_input: AgentInput) -> dict:
    """
    Calls Groq to parse the user's raw intent into structured form.
    Returns a dict matching the JSON shape above.
    """
    user_message = (
        f"Room image URL: {agent_input.image_url}\n"
        f"User request: {agent_input.user_intent}\n"
        f"Budget (INR): {agent_input.budget_inr}"
    )

    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


async def run_agents_parallel(agent_input: AgentInput):
    """
    Dispatches all 4 agents concurrently using asyncio.
    Each agent is imported lazily to avoid circular imports.
    """
    from agents.vision_agent import run as vision_run
    from agents.style_agent import run as style_run
    from agents.layout_agent import run as layout_run
    from agents.commerce_agent import run as commerce_run

    vision_task = asyncio.to_thread(vision_run, agent_input)
    style_task = asyncio.to_thread(style_run, agent_input)
    commerce_task = asyncio.to_thread(commerce_run, agent_input)

    # Yield control to let them start
    vision_out, style_out, commerce_out = await asyncio.gather(
        vision_task, style_task, commerce_task
    )

    # Layout Agent depends on Vision Agent's output!
    layout_out = await asyncio.to_thread(layout_run, agent_input, vision_out)

    return {
        "vision": vision_out,
        "style": style_out,
        "layout": layout_out,
        "commerce": commerce_out,
    }


async def orchestrate(agent_input: AgentInput) -> dict:
    """
    Full orchestration pipeline:
    1. Parse intent with Groq
    2. Run all 4 agents in parallel
    3. Return merged outputs for the aggregator
    """
    parsed = parse_intent(agent_input)
    print(f"[Orchestrator] Parsed intent: {parsed['parsed_intent']}")
    print(f"[Orchestrator] Style hint: {parsed['style_hint']}")

    agent_outputs = await run_agents_parallel(agent_input)

    return {
        "parsed_intent": parsed,
        "agent_outputs": agent_outputs,
    }
