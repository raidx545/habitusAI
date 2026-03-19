"""
style_agent.py — Agent 2: Style Agent
Fast style classification: keyword matching on user intent + Groq fallback.

Runs on Mac (CPU). No CLIP model download needed.
Install: pip install groq pydantic python-dotenv
"""

import sys, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path('/content/drive/MyDrive/HabitusAI')
sys.path.insert(0, str(ROOT))
from contracts import AgentInput, StyleOutput
from groq import Groq

# ── Style database ─────────────────────────────────────────────────────────────
STYLES = {
    "minimalist": {
        "keywords": "clean lines, white walls, neutral palette, uncluttered, functional furniture, negative space",
        "triggers": ["minimal", "minimalist", "clean", "simple", "clutter-free", "sparse"],
    },
    "scandinavian": {
        "keywords": "light oak wood, white walls, linen textiles, hygge, warm ambient lighting, cozy",
        "triggers": ["scandinavian", "scandi", "nordic", "hygge", "danish", "cozy", "warm"],
    },
    "japandi": {
        "keywords": "wabi-sabi, natural wood, muted earth tones, paper lanterns, tatami, zen simplicity",
        "triggers": ["japandi", "japanese", "zen", "wabi", "sabi", "japan", "tatami"],
    },
    "maximalist": {
        "keywords": "jewel tones, ornate details, layered textiles, bold patterns, gallery wall, eclectic",
        "triggers": ["maximalist", "bold", "colorful", "vibrant", "eclectic", "ornate", "layered"],
    },
    "industrial": {
        "keywords": "exposed brick, raw concrete, metal accents, Edison bulbs, dark palette, warehouse",
        "triggers": ["industrial", "loft", "urban", "brick", "concrete", "metal", "warehouse"],
    },
    "bohemian": {
        "keywords": "warm earth tones, macramé, rattan furniture, layered rugs, plants, free-spirited",
        "triggers": ["bohemian", "boho", "rattan", "macrame", "earthy", "plants", "hippie"],
    },
    "coastal": {
        "keywords": "ocean blues, whitewashed wood, natural linen, seagrass, driftwood, breezy",
        "triggers": ["coastal", "beach", "ocean", "nautical", "seaside", "marine", "blue"],
    },
    "contemporary": {
        "keywords": "sleek surfaces, neutral base, mixed materials, clean geometry, sophisticated",
        "triggers": ["contemporary", "modern", "sleek", "sophisticated", "current", "stylish"],
    },
}


def _keyword_match(user_intent: str) -> tuple[str | None, float]:
    """
    Fast O(1) keyword scan of user_intent.
    Returns (style_label, confidence) or (None, 0).
    """
    text = user_intent.lower()
    scores: dict[str, int] = {}
    for style, data in STYLES.items():
        for trigger in data["triggers"]:
            if trigger in text:
                scores[style] = scores.get(style, 0) + 1

    if not scores:
        return None, 0.0

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = round(scores[best] / total, 3)
    return best, confidence


def _groq_classify(user_intent: str, image_url: str) -> tuple[str, float]:
    """
    Groq-powered style classification — used when keyword matching is ambiguous.
    """
    client = Groq()
    style_list = ", ".join(STYLES.keys())

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an interior design style expert. "
                    f"Classify the user's request into exactly ONE of: {style_list}. "
                    f"Return ONLY a JSON object: {{\"style\": \"<label>\", \"confidence\": <0.0-1.0>}}"
                ),
            },
            {
                "role": "user",
                "content": f"Room image: {image_url}\nUser request: {user_intent}",
            },
        ],
        temperature=0.1,
        max_tokens=64,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    style = result.get("style", "contemporary")
    confidence = float(result.get("confidence", 0.7))

    # Validate style label
    if style not in STYLES:
        style = "contemporary"

    return style, round(confidence, 3)


def run(agent_input: AgentInput) -> StyleOutput:
    """
    Main entry point — called by the orchestrator.
    Fast path: keyword match → returns instantly.
    Slow path: Groq classification → ~1 API call.
    """
    print(f"[StyleAgent] Classifying style for: '{agent_input.user_intent[:60]}'")

    # Fast path — keyword matching
    style, confidence = _keyword_match(agent_input.user_intent)

    if style and confidence >= 0.5:
        print(f"[StyleAgent] Keyword match → '{style}' (confidence: {confidence})")
    else:
        # Groq fallback for ambiguous intent
        print("[StyleAgent] Intent ambiguous — falling back to Groq classification...")
        style, confidence = _groq_classify(
            agent_input.user_intent, agent_input.image_url
        )
        print(f"[StyleAgent] Groq classified → '{style}' (confidence: {confidence})")

    return StyleOutput(
        style_label=style,
        style_keywords=STYLES[style]["keywords"],
        clip_vector=[],   # not used by aggregator; saves ~900MB CLIP model
        confidence=confidence,
    )


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("Make it Scandinavian and cozy", "explicit keyword"),
        ("I want something modern and sleek", "explicit keyword"),
        ("Transform into a beautiful room", "ambiguous → Groq"),
    ]
    for intent, note in tests:
        print(f"\n[Test — {note}]")
        inp = AgentInput(
            image_url="https://images.unsplash.com/photo-1618220179428-22790b461013?w=640",
            user_intent=intent,
            budget_inr=45000,
        )
        out = run(inp)
        print(f"  Style    : {out.style_label}")
        print(f"  Keywords : {out.style_keywords}")
        print(f"  Confidence: {out.confidence}")
