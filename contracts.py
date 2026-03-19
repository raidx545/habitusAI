# contracts.py — both teammates commit to these exactly
# DO NOT change these shapes without telling the other person

from pydantic import BaseModel
from typing import List, Optional

# ── INPUT (same for all agents) ──────────────────────────
class AgentInput(BaseModel):
    image_url: str          # public URL of room image
    user_intent: str        # raw user text
    budget_inr: Optional[int] = 50000

# ── AGENT 1 OUTPUT — Vision (Raj) ────────────────────────
class BoundingBox(BaseModel):
    label: str
    confidence: float
    bbox: List[int]         # [x1, y1, x2, y2]

class VisionOutput(BaseModel):
    depth_map_path: str     # local path or URL to depth.png
    objects: List[BoundingBox]
    room_dims_estimate: dict  # {"w_m": 5.2, "h_m": 3.1, "d_m": 4.8}

# ── AGENT 2 OUTPUT — Style (Raj) ─────────────────────────
class StyleOutput(BaseModel):
    style_label: str        # e.g. "maximalist"
    style_keywords: str     # e.g. "jewel tones, ornate, layered"
    clip_vector: List[float]  # 512-dim embedding
    confidence: float

# ── AGENT 3 OUTPUT — Layout (friend) ─────────────────────
class FurnitureZone(BaseModel):
    item: str               # "king_bed"
    wall: str               # "back_center"
    position_pct: dict      # {"x": 0.5, "y": 0.1} % of room

class LayoutOutput(BaseModel):
    floor_plan_path: str    # path to floor plan image
    zones: List[FurnitureZone]
    clearance_ok: bool

# ── AGENT 4 OUTPUT — Commerce (friend) ───────────────────
class IKEAProduct(BaseModel):
    name: str
    category: str           # "bed", "sofa", "fan"
    price_inr: int
    url: str
    image_url: str

class CommerceOutput(BaseModel):
    products: List[IKEAProduct]
    total_price_inr: int
    within_budget: bool
"""```

Both of you import from this single file. Claude's aggregator also imports from it. Nobody invents their own format.

---"""

## How to work independently without blocking each other

### Git structure
"""```
interior-ai/
├── contracts.py          ← shared, never edit without discussion
├── orchestrator/         ← Claude aggregator (decide who owns this)
│   └── aggregator.py
├── agents/
│   ├── vision_agent.py   ← Raj
│   ├── style_agent.py    ← Raj
│   ├── layout_agent.py   ← friend
│   └── commerce_agent.py ← friend
├── generation/
│   └── sdxl_pipeline.py  ← whoever finishes their agents first
├── api/
│   └── main.py           ← FastAPI — whoever owns the backend
└── frontend/             ← Next.js
'''"""