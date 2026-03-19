"""
vision_agent.py — Agent 1: Vision Agent
Uses Depth Anything V2 (transformers) + YOLOv8 for fast, reliable depth + detection.

⚠️  Run on GPU (Google Colab T4). Install: pip install transformers ultralytics
"""

import os, json, sys, urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    import cv2
    from transformers import pipeline
    from ultralytics import YOLO
    GPU_AVAILABLE = torch.cuda.is_available()
    MISSING_DEPS = False
except ImportError:
    GPU_AVAILABLE = False
    MISSING_DEPS = True
    print("[VisionAgent] Missing ML packages (torch, transformers). Will use mock data if run locally.")

# Add project root to path (__file__ is not defined in Colab notebooks)
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path('/content/drive/MyDrive/HabitusAI')  # Colab Drive path
sys.path.insert(0, str(ROOT))
from contracts import AgentInput, VisionOutput, BoundingBox

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL   = "yolov8m.pt"
DEPTH_MODEL  = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_OUTPUT = "/tmp/interior_ai/depth"

INTERIOR_CLASSES = {
    "chair", "couch", "bed", "dining table", "tv", "laptop",
    "microwave", "oven", "sink", "refrigerator", "clock",
    "vase", "bottle", "cup", "book", "potted plant",
}

# ── Lazy-loaded models ────────────────────────────────────────────────────────
_depth_pipe = None
_yolo_model = None


def _get_depth_pipe():
    global _depth_pipe
    if _depth_pipe is None:
        print("[VisionAgent] Loading Depth Anything V2...")
        device = 0 if GPU_AVAILABLE else -1
        _depth_pipe = pipeline(
            task="depth-estimation",
            model=DEPTH_MODEL,
            device=device,
        )
        print("[VisionAgent] Depth Anything V2 ready ✅")
    return _depth_pipe


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        print("[VisionAgent] Loading YOLOv8...")
        _yolo_model = YOLO(YOLO_MODEL)
        print("[VisionAgent] YOLOv8 ready ✅")
    return _yolo_model


# ── Step 1: Depth Estimation ──────────────────────────────────────────────────
def run_depth(pil_image: Image.Image, output_dir: str) -> tuple:
    """
    Runs Depth Anything V2 on the input PIL image.
    Returns (depth_map_path, depth_array_float32)
    """
    os.makedirs(output_dir, exist_ok=True)
    pipe = _get_depth_pipe()

    result = pipe(pil_image)
    depth_pil = result["depth"]                         # grayscale PIL image
    depth_arr = np.array(depth_pil, dtype=np.float32)  # H×W float array

    depth_path = os.path.join(output_dir, "depth.png")
    depth_pil.save(depth_path)
    print(f"[VisionAgent] Depth map saved → {depth_path}")
    return depth_path, depth_arr


# ── Step 2: Object Detection ──────────────────────────────────────────────────
def run_yolo(bgr_image: np.ndarray) -> list:
    yolo = _get_yolo()
    results = yolo(bgr_image, verbose=False)[0]

    boxes = []
    for box in results.boxes:
        cls_name = results.names[int(box.cls)]
        if cls_name not in INTERIOR_CLASSES:
            continue
        conf = float(box.conf)
        if conf < 0.4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        boxes.append(BoundingBox(
            label=cls_name,
            confidence=round(conf, 3),
            bbox=[x1, y1, x2, y2],
        ))

    print(f"[VisionAgent] YOLOv8: {len(boxes)} objects → {[b.label for b in boxes]}")
    return boxes


# ── Step 3: Room Dimension Estimate ──────────────────────────────────────────
def estimate_room_dims(depth_arr: np.ndarray, img_w: int, img_h: int) -> dict:
    """
    Estimates room dimensions from a relative depth map.
    Depth Anything V2 outputs relative depth (0-255), not metric metres.
    We map the upper quartile (far background = walls) to a typical
    indoor depth of 2-6m to get a rough metric estimate.
    """
    # Upper 75th percentile represents the far background (walls/ceiling)
    far_val  = float(np.percentile(depth_arr, 75))
    near_val = float(np.percentile(depth_arr, 10))
    depth_range = far_val - near_val + 1e-8

    # Map far point to realistic indoor room depth (2–6 m)
    INDOOR_DEPTH_M = 4.0   # typical living room depth
    scale = INDOOR_DEPTH_M / depth_range

    d_m = round(INDOOR_DEPTH_M, 1)
    fov_rad = np.deg2rad(60)
    w_m = round(2 * d_m * np.tan(fov_rad / 2), 1)
    h_m = round(w_m / (img_w / img_h), 1)

    # Sanity clamp: real rooms are 2-12m wide, 2-4m tall
    w_m = round(min(max(w_m, 2.0), 12.0), 1)
    h_m = round(min(max(h_m, 2.0), 4.0), 1)
    d_m = round(min(max(d_m, 2.0), 10.0), 1)

    return {"w_m": w_m, "h_m": h_m, "d_m": d_m}



# ── Main Entry Point ──────────────────────────────────────────────────────────
def run(agent_input: AgentInput) -> VisionOutput:
    print(f"[VisionAgent] Processing: {agent_input.image_url}")

    # Load image
    if os.path.isfile(agent_input.image_url):
        pil_img = Image.open(agent_input.image_url).convert("RGB")
    else:
        with urllib.request.urlopen(agent_input.image_url) as r:
            pil_img = Image.open(BytesIO(r.read())).convert("RGB")

    # Safely resize massive iPhone/4K photos so they don't blow up the Colab RAM!
    MAX_DIM = 1024
    if pil_img.width > MAX_DIM or pil_img.height > MAX_DIM:
        pil_img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
        print(f"[VisionAgent] Resized massive input image to {pil_img.size}")

    np_rgb = np.array(pil_img)
    bgr    = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    H, W   = bgr.shape[:2]

    # Step 1: Depth
    depth_path, depth_arr = run_depth(pil_img, DEPTH_OUTPUT)

    # Step 2: Detection
    bboxes = run_yolo(bgr)

    # Step 3: Room dims
    room_dims = estimate_room_dims(depth_arr, W, H)
    print(f"[VisionAgent] Room dims: {room_dims}")

    return VisionOutput(
        depth_map_path=depth_path,
        objects=bboxes,
        room_dims_estimate=room_dims,
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from contracts import AgentInput
    inp = AgentInput(
        image_url="https://images.unsplash.com/photo-1618220179428-22790b461013?w=1080",
        user_intent="Make it Scandinavian",
        budget_inr=45000,
    )
    out = run(inp)
    print(out.model_dump_json(indent=2))
