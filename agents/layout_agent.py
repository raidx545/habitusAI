"""
layout_agent.py — Agent 3: Layout Agent
Converts Vision Agent bounding boxes → furniture zones + floor plan image.

Approach:
  - Projects image-space bounding boxes into floor-plan (top-down) space
  - Uses Shapely to check clearance between furniture polygons
  - Draws a bird's-eye floor plan with Pillow

Runs on Mac (CPU). No GPU needed.
Install: pip install shapely Pillow numpy
"""

import os, sys
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from shapely.geometry import box as shapely_box
    from shapely.ops import unary_union
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False
    print("[LayoutAgent] shapely not installed — clearance check disabled")

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path('/content/drive/MyDrive/HabitusAI')
sys.path.insert(0, str(ROOT))
from contracts import AgentInput, VisionOutput, FurnitureZone, LayoutOutput

# ── Config ────────────────────────────────────────────────────────────────────
FLOOR_PLAN_PATH = "/tmp/interior_ai/floor_plan.png"
FLOOR_PLAN_W    = 600   # pixels
FLOOR_PLAN_H    = 600

# Minimum clearance between furniture in floor-plan pixels
MIN_CLEARANCE_PX = 30

# Colors per furniture category (for the floor plan drawing)
CATEGORY_COLORS = {
    "couch":        "#E57373",  # red
    "chair":        "#F06292",  # pink
    "bed":          "#9575CD",  # purple
    "dining table": "#4DB6AC",  # teal
    "tv":           "#64B5F6",  # blue
    "potted plant": "#81C784",  # green
    "vase":         "#A5D6A7",  # light green
    "book":         "#FFD54F",  # yellow
    "laptop":       "#4DD0E1",  # cyan
    "sink":         "#90A4AE",  # grey
    "refrigerator": "#B0BEC5",  # grey-blue
    "clock":        "#BCAAA4",  # brown
    "_default":     "#CFD8DC",  # blue-grey
}


def _bbox_to_zone(bbox: list, img_w: int, img_h: int, label: str) -> FurnitureZone:
    """
    Projects an image-space bounding box into floor-plan percentage coordinates.

    Image-space intuition:
      - x-center maps directly to floor x (left/right in the room)
      - y in the image maps to depth (y near top = far from camera = back wall)

    Returns a FurnitureZone with position_pct and wall assignment.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2   # horizontal center in image
    cy = (y1 + y2) / 2   # vertical center in image

    # Normalise to 0-1
    x_pct = round(cx / img_w, 3)
    y_pct = round(cy / img_h, 3)   # 0 = top of image = back wall

    # Wall assignment heuristic
    if y_pct < 0.35:
        wall = "back"
    elif y_pct > 0.70:
        wall = "front"
    elif x_pct < 0.25:
        wall = "left"
    elif x_pct > 0.75:
        wall = "right"
    else:
        wall = "center"

    # Item name: label with underscores
    item = label.replace(" ", "_")

    return FurnitureZone(
        item=item,
        wall=wall,
        position_pct={"x": x_pct, "y": y_pct},
    )


def _draw_floor_plan(
    zones: list[FurnitureZone],
    img_w: int,
    img_h: int,
    room_dims: dict,
    output_path: str,
) -> None:
    """
    Draws a simple top-down floor plan with labeled furniture rectangles.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create canvas
    img = Image.new("RGB", (FLOOR_PLAN_W, FLOOR_PLAN_H), "#F5F5F5")
    draw = ImageDraw.Draw(img)

    # Draw room outline
    MARGIN = 40
    room_rect = [MARGIN, MARGIN, FLOOR_PLAN_W - MARGIN, FLOOR_PLAN_H - MARGIN]
    draw.rectangle(room_rect, outline="#424242", width=3)

    # Labels: "BACK WALL" / "FRONT"
    draw.text((FLOOR_PLAN_W // 2, MARGIN // 2), "BACK WALL",
              fill="#757575", anchor="mm")
    draw.text((FLOOR_PLAN_W // 2, FLOOR_PLAN_H - MARGIN // 2), "FRONT",
              fill="#757575", anchor="mm")
    draw.text((MARGIN // 2, FLOOR_PLAN_H // 2), "L",
              fill="#757575", anchor="mm")
    draw.text((FLOOR_PLAN_W - MARGIN // 2, FLOOR_PLAN_H // 2), "R",
              fill="#757575", anchor="mm")

    # Draw room dimensions text
    w_m = room_dims.get("w_m", "?")
    d_m = room_dims.get("d_m", "?")
    draw.text((FLOOR_PLAN_W // 2, MARGIN + 12),
              f"{w_m}m wide × {d_m}m deep",
              fill="#9E9E9E", anchor="mm")

    # Inner room bounds for positioning furniture
    inner_x0, inner_y0 = MARGIN + 4, MARGIN + 30
    inner_w = FLOOR_PLAN_W - 2 * MARGIN - 8
    inner_h = FLOOR_PLAN_H - 2 * MARGIN - 34

    # Size of each furniture rectangle (proportional to inner room)
    FW = max(30, inner_w // 8)
    FH = max(20, inner_h // 8)

    for zone in zones:
        x_pct = zone.position_pct["x"]
        y_pct = zone.position_pct["y"]

        # Map percentages to pixel coords within inner room
        cx = int(inner_x0 + x_pct * inner_w)
        cy = int(inner_y0 + y_pct * inner_h)

        x0, y0 = cx - FW // 2, cy - FH // 2
        x1, y1 = cx + FW // 2, cy + FH // 2

        color = CATEGORY_COLORS.get(zone.item.replace("_", " "),
                                    CATEGORY_COLORS["_default"])

        draw.rectangle([x0, y0, x1, y1], fill=color, outline="#424242", width=1)
        draw.text((cx, cy), zone.item.replace("_", "\n"),
                  fill="#212121", anchor="mm", font_size=9)

    img.save(output_path)
    print(f"[LayoutAgent] Floor plan saved → {output_path}")


def _check_clearance(zones: list[FurnitureZone]) -> bool:
    """
    Uses Shapely to check whether any two furniture zones overlap.
    FW/FH are estimated rectangle sizes in % space (0-1).
    """
    if not SHAPELY_OK or len(zones) < 2:
        return True

    FW_PCT = 0.12
    FH_PCT = 0.12

    polygons = []
    for zone in zones:
        x = zone.position_pct["x"]
        y = zone.position_pct["y"]
        poly = shapely_box(x - FW_PCT/2, y - FH_PCT/2,
                           x + FW_PCT/2, y + FH_PCT/2)
        polygons.append(poly)

    # Check pairwise for overlap (with MIN_CLEARANCE buffer in % space)
    CLEARANCE_PCT = MIN_CLEARANCE_PX / FLOOR_PLAN_W
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            buffered = polygons[i].buffer(CLEARANCE_PCT)
            if buffered.intersects(polygons[j]):
                print(f"[LayoutAgent] Clearance issue: {zones[i].item} ↔ {zones[j].item}")
                return False
    return True


# ── Main Entry Point ──────────────────────────────────────────────────────────
def run(agent_input: AgentInput, vision_output: VisionOutput) -> LayoutOutput:
    """
    Takes AgentInput + VisionOutput (from Agent 1).
    Returns LayoutOutput with floor plan + furniture zones.
    """
    print(f"[LayoutAgent] Processing {len(vision_output.objects)} detected objects...")

    if not vision_output.objects:
        print("[LayoutAgent] No objects detected — returning empty layout")
        return LayoutOutput(
            floor_plan_path=FLOOR_PLAN_PATH,
            zones=[],
            clearance_ok=True,
        )

    # We need image dimensions to normalise — estimate from URL or use a default
    # (Vision Agent bboxes use actual image pixels so we need the original size)
    # Estimate from the largest bbox coordinate
    all_coords = [c for obj in vision_output.objects for c in obj.bbox]
    img_w = max(all_coords[0::2] + [1080])  # x coords → width
    img_h = max(all_coords[1::2] + [1080])  # y coords → height

    # Build furniture zones from bounding boxes
    zones: list[FurnitureZone] = []
    seen_items: dict[str, int] = {}

    for obj in vision_output.objects:
        # Deduplicate: rename duplicates as chair_1, chair_2 etc.
        base = obj.label.replace(" ", "_")
        seen_items[base] = seen_items.get(base, 0) + 1
        label = obj.label if seen_items[base] == 1 else \
                f"{obj.label} {seen_items[base]}"

        zone = _bbox_to_zone(obj.bbox, img_w, img_h, label)
        zones.append(zone)

    # Draw floor plan
    _draw_floor_plan(zones, img_w, img_h,
                     vision_output.room_dims_estimate, FLOOR_PLAN_PATH)

    # Check clearance
    clearance_ok = _check_clearance(zones)
    print(f"[LayoutAgent] Clearance OK: {clearance_ok}")
    print(f"[LayoutAgent] Zones: {[z.item for z in zones]}")

    return LayoutOutput(
        floor_plan_path=FLOOR_PLAN_PATH,
        zones=zones,
        clearance_ok=clearance_ok,
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from contracts import AgentInput, VisionOutput, BoundingBox

    mock_vision = VisionOutput(
        depth_map_path="/tmp/depth.png",
        objects=[
            BoundingBox(label="couch",        confidence=0.93, bbox=[583,1113,922,1390]),
            BoundingBox(label="chair",         confidence=0.87, bbox=[200, 900,480,1200]),
            BoundingBox(label="potted plant",  confidence=0.91, bbox=[1,   794, 323,1318]),
            BoundingBox(label="vase",          confidence=0.88, bbox=[672, 994, 725,1095]),
            BoundingBox(label="dining table",  confidence=0.76, bbox=[400, 300, 700, 500]),
        ],
        room_dims_estimate={"w_m": 5.2, "h_m": 3.0, "d_m": 4.3},
    )

    inp = AgentInput(
        image_url="https://images.unsplash.com/photo-1618220179428-22790b461013?w=1080",
        user_intent="Make it Scandinavian",
        budget_inr=45000,
    )

    result = run(inp, mock_vision)
    print(f"\nFloor plan: {result.floor_plan_path}")
    print(f"Clearance OK: {result.clearance_ok}")
    for z in result.zones:
        print(f"  {z.item:20s} wall={z.wall:8s} pos={z.position_pct}")
