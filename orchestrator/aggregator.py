"""
aggregator.py — Groq-powered Aggregator
Merges all 4 agent outputs into a structured SDXL prompt and final response.
"""

import json
from groq import Groq
from contracts import VisionOutput, StyleOutput, LayoutOutput, CommerceOutput

# ── Groq config ───────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"

def _get_client() -> Groq:
    """Lazy init — only fails if GROQ_API_KEY missing when actually called."""
    return Groq()  # reads GROQ_API_KEY from env

SYSTEM_PROMPT = """You are an expert interior design AI and Stable Diffusion prompt engineer.

You will receive structured data from 4 specialized agents:
- Vision Agent: room dimensions, existing objects, depth map info
- Style Agent: detected/target style label and keywords
- Layout Agent: furniture placement zones and floor plan info
- Commerce Agent: product list with IKEA items and prices

Your job is to synthesize all this data and return a JSON with two keys:
1. "sdxl_prompt": A rich, photorealistic Stable Diffusion XL prompt (150-250 words).
   - Describe the redesigned room in vivid detail using the style keywords.
   - Mention specific furniture placements from the layout agent.
   - Include lighting, textures, materials, atmosphere.
   - CRITICAL: You MUST explicitly include the keywords "highly organized, decluttered, immaculate, neat, architectural, minimalist organization" in the prompt.
   - Always end with: "photorealistic, 8K, architectural photography, interior design magazine"
   - Do NOT include negative prompts here.

2. "negative_prompt": A compact comma-separated list of things to avoid.
   Standard negatives: "cartoon, illustration, painting, 3d render, blurry, low quality, distorted, watermark"

3. "design_summary": A friendly 2-3 sentence summary for the user explaining what was designed and why.

Return ONLY raw JSON. No markdown, no extra text."""


def build_sdxl_prompt(
    parsed_intent: dict,
    vision: VisionOutput,
    style: StyleOutput,
    layout: LayoutOutput,
    commerce: CommerceOutput,
) -> dict:
    """
    Calls Groq to aggregate all agent outputs into a structured SDXL prompt.

    Args:
        parsed_intent: Output from orchestrator.parse_intent()
        vision: VisionOutput from Vision Agent
        style: StyleOutput from Style Agent
        layout: LayoutOutput from Layout Agent
        commerce: CommerceOutput from Commerce Agent

    Returns:
        dict with keys: sdxl_prompt, negative_prompt, design_summary
    """
    # Build a structured context for Groq
    context = {
        "user_intent": parsed_intent,
        "vision": {
            "room_dimensions": vision.room_dims_estimate,
            "detected_objects": [
                {"label": obj.label, "confidence": round(obj.confidence, 2)}
                for obj in vision.objects
            ],
            "depth_map_available": True,
        },
        "style": {
            "detected_style": style.style_label,
            "style_keywords": style.style_keywords,
            "confidence": round(style.confidence, 2),
        },
        "layout": {
            "furniture_zones": [
                {"item": z.item, "wall": z.wall, "position": z.position_pct}
                for z in layout.zones
            ],
            "clearance_validated": layout.clearance_ok,
        },
        "commerce": {
            "total_budget_inr": commerce.total_price_inr,
            "within_budget": commerce.within_budget,
            "key_products": [
                {"name": p.name, "category": p.category, "price_inr": p.price_inr}
                for p in commerce.products[:5]  # top 5 items for context
            ],
        },
    }

    user_message = f"Here is the aggregated data from all agents:\n{json.dumps(context, indent=2)}"

    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,   # slightly creative for prompt writing
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Validate keys
    assert "sdxl_prompt" in result, "Groq response missing 'sdxl_prompt'"
    assert "negative_prompt" in result, "Groq response missing 'negative_prompt'"
    assert "design_summary" in result, "Groq response missing 'design_summary'"

    return result


def aggregate(orchestrator_output: dict) -> dict:
    """
    Top-level aggregation function.
    Takes the full orchestrator output and returns SDXL-ready data.

    Args:
        orchestrator_output: Result from orchestrator.orchestrate()
            {
                "parsed_intent": {...},
                "agent_outputs": {
                    "vision": VisionOutput,
                    "style": StyleOutput,
                    "layout": LayoutOutput,
                    "commerce": CommerceOutput,
                }
            }

    Returns:
        dict with: sdxl_prompt, negative_prompt, design_summary,
                   depth_map_path, floor_plan_path, products
    """
    parsed_intent = orchestrator_output["parsed_intent"]
    agents = orchestrator_output["agent_outputs"]

    vision: VisionOutput = agents["vision"]
    style: StyleOutput = agents["style"]
    layout: LayoutOutput = agents["layout"]
    commerce: CommerceOutput = agents["commerce"]

    print("[Aggregator] Calling Groq to build SDXL prompt...")
    groq_result = build_sdxl_prompt(parsed_intent, vision, style, layout, commerce)

    print(f"[Aggregator] Design summary: {groq_result['design_summary']}")

    return {
        # SDXL inputs
        "sdxl_prompt": groq_result["sdxl_prompt"],
        "negative_prompt": groq_result["negative_prompt"],
        "depth_map_path": vision.depth_map_path,     # fed to ControlNet

        # User-facing outputs
        "design_summary": groq_result["design_summary"],
        "floor_plan_path": layout.floor_plan_path,
        "products": [p.dict() for p in commerce.products],
        "within_budget": commerce.within_budget,
        "total_price_inr": commerce.total_price_inr,
    }
