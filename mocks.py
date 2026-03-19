"""
mocks.py — Fake agent outputs for testing the orchestrator pipeline
Use these when the real agents haven't been implemented yet.
Both teammates can test their orchestrator/aggregator independently.
"""

from contracts import (
    AgentInput,
    BoundingBox,
    VisionOutput,
    StyleOutput,
    LayoutOutput,
    FurnitureZone,
    CommerceOutput,
    IKEAProduct,
)

# ── Sample input ──────────────────────────────────────────────────────────────
MOCK_INPUT = AgentInput(
    image_url="https://images.unsplash.com/photo-1618220179428-22790b461013",
    user_intent="Make it Scandinavian and cozy, I want warm lighting and light wood furniture",
    budget_inr=45000,
)

# ── Mock Vision Output ────────────────────────────────────────────────────────
MOCK_VISION = VisionOutput(
    depth_map_path="/tmp/mock_depth.png",
    objects=[
        BoundingBox(label="sofa", confidence=0.94, bbox=[120, 200, 480, 420]),
        BoundingBox(label="coffee_table", confidence=0.88, bbox=[200, 350, 380, 430]),
        BoundingBox(label="window", confidence=0.97, bbox=[0, 80, 150, 350]),
        BoundingBox(label="floor_lamp", confidence=0.76, bbox=[490, 150, 540, 400]),
    ],
    room_dims_estimate={"w_m": 4.5, "h_m": 2.8, "d_m": 5.2},
)

# ── Mock Style Output ─────────────────────────────────────────────────────────
MOCK_STYLE = StyleOutput(
    style_label="scandinavian",
    style_keywords="light oak wood, white walls, linen textiles, hygge, warm ambient lighting, minimal clutter",
    clip_vector=[0.1] * 512,   # placeholder, not used by aggregator
    confidence=0.91,
)

# ── Mock Layout Output ────────────────────────────────────────────────────────
MOCK_LAYOUT = LayoutOutput(
    floor_plan_path="/tmp/mock_floor_plan.png",
    zones=[
        FurnitureZone(item="2_seater_sofa", wall="back_center", position_pct={"x": 0.5, "y": 0.15}),
        FurnitureZone(item="coffee_table", wall="center", position_pct={"x": 0.5, "y": 0.42}),
        FurnitureZone(item="floor_lamp", wall="back_right", position_pct={"x": 0.85, "y": 0.18}),
        FurnitureZone(item="side_table", wall="back_left", position_pct={"x": 0.15, "y": 0.2}),
    ],
    clearance_ok=True,
)

# ── Mock Commerce Output ──────────────────────────────────────────────────────
MOCK_COMMERCE = CommerceOutput(
    products=[
        IKEAProduct(
            name="KIVIK 2-seat sofa",
            category="sofa",
            price_inr=32000,
            url="https://www.ikea.com/in/en/p/kivik-2-seat-sofa/",
            image_url="https://www.ikea.com/in/en/images/products/kivik-2-seat-sofa.jpg",
        ),
        IKEAProduct(
            name="HEMNES coffee table",
            category="coffee_table",
            price_inr=8500,
            url="https://www.ikea.com/in/en/p/hemnes-coffee-table/",
            image_url="https://www.ikea.com/in/en/images/products/hemnes-coffee-table.jpg",
        ),
        IKEAProduct(
            name="HEKTAR floor lamp",
            category="lamp",
            price_inr=4500,
            url="https://www.ikea.com/in/en/p/hektar-floor-lamp/",
            image_url="https://www.ikea.com/in/en/images/products/hektar-floor-lamp.jpg",
        ),
    ],
    total_price_inr=45000,
    within_budget=True,
)


def get_mock_agent_outputs() -> dict:
    """Returns a mock orchestrator output dict ready for the aggregator."""
    return {
        "vision": MOCK_VISION,
        "style": MOCK_STYLE,
        "layout": MOCK_LAYOUT,
        "commerce": MOCK_COMMERCE,
    }
