# Vision Agent — Google Colab Setup Guide
### Your project path: `/content/Mydrive/HabitusAI/`

---

## Step 1 — Enable GPU

Runtime → Change runtime type → **T4 GPU** → Save

---

## Step 2 — Mount Google Drive (Cell 1)

```python
# Cell 1
from google.colab import drive
drive.mount('/content/Mydrive')

import sys
# Add project root to Python path so 'contracts' is importable
sys.path.insert(0, '/content/Mydrive/HabitusAI')
sys.path.insert(0, '/content/Mydrive/HabitusAI/Agents')

print("✅ Drive mounted and paths set")
```

---

## Step 3 — Install dependencies (Cell 2)

```python
# Cell 2 — Install all Vision Agent GPU dependencies
!pip install ultralytics segment-anything torch torchvision \
             Pillow numpy opencv-python-headless pydantic groq -q

# ZoeDepth from source
!git clone https://github.com/isl-org/ZoeDepth.git /content/ZoeDepth
%cd /content/ZoeDepth
!pip install -e . -q
%cd /content

import torch
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
```

---

## Step 4 — Run the Vision Agent (Cell 3)

```python
# Cell 3 — Run Vision Agent
import sys
sys.path.insert(0, '/content/Mydrive/HabitusAI')     # for contracts.py
sys.path.insert(0, '/content/Mydrive/HabitusAI/Agents')  # for vision_agent.py
sys.path.insert(0, '/content/ZoeDepth')              # for zoedepth module

from contracts import AgentInput
from vision_agent import run

test_input = AgentInput(
    image_url="https://images.unsplash.com/photo-1618220179428-22790b461013?w=1080",
    user_intent="Make it Scandinavian and cozy",
    budget_inr=45000,
)

result = run(test_input)
print(result.model_dump_json(indent=2))
```

---

## Step 5 — Visualise depth map (Cell 4)

```python
# Cell 4 — Show original + depth map side by side
from PIL import Image
import matplotlib.pyplot as plt
import urllib.request
from io import BytesIO

with urllib.request.urlopen(test_input.image_url) as r:
    original = Image.open(BytesIO(r.read()))

depth = Image.open(result.depth_map_path)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(original)
axes[0].set_title("Original Room")
axes[0].axis("off")
axes[1].imshow(depth, cmap="inferno")
axes[1].set_title("ZoeDepth — Depth Map")
axes[1].axis("off")
plt.tight_layout()
plt.show()

print(f"Detected: {[o.label for o in result.objects]}")
print(f"Room dims: {result.room_dims_estimate}")
```

---

## Step 6 — Expose as API via ngrok (Cell 5)
*Only needed when your Mac FastAPI needs to call this agent*

```python
# Cell 5 — Serve Vision Agent as REST API
!pip install fastapi uvicorn pyngrok nest-asyncio -q

import nest_asyncio, threading
nest_asyncio.apply()

from fastapi import FastAPI
from pyngrok import ngrok
import uvicorn

app = FastAPI()

@app.post("/vision")
def vision_endpoint(payload: dict):
    from contracts import AgentInput
    from vision_agent import run
    result = run(AgentInput(**payload))
    return result.model_dump()

public_url = ngrok.connect(8000)
print(f"✅ Vision Agent live at: {public_url}/vision")
print(f"→ Add to your Mac .env: GPU_VISION_URL={public_url}")

threading.Thread(
    target=uvicorn.run,
    args=(app,),
    kwargs={"host": "0.0.0.0", "port": 8000}
).start()
```

---

## File placement on Drive

```
MyDrive/HabitusAI/
├── contracts.py              ← copy from your Mac project
├── Agents/
│   └── vision_agent.py      ← copy from your Mac project
```

## Timing reference

| Step | First run | After caching |
|---|---|---|
| YOLOv8 weights | ~30 sec | instant |
| SAM checkpoint (2.4 GB) | ~5 min | instant |
| ZoeDepth weights | ~2 min | instant |
| Inference per image | — | ~25 sec |

> **T4 VRAM:** ZoeDepth 2 GB + YOLOv8 1 GB + SAM 4 GB ≈ 7 GB / 15 GB available ✅


1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. **Runtime → Change runtime type → T4 GPU → Save**

---

## Step 2 — Install dependencies (Cell 1)

Paste this in the first cell and run it:

```python
# Cell 1 — Install all Vision Agent dependencies
!pip install ultralytics segment-anything torch torchvision \
             Pillow numpy opencv-python-headless groq pydantic -q

# ZoeDepth — install from official repo
!git clone https://github.com/isl-org/ZoeDepth.git
%cd ZoeDepth
!pip install -e . -q
%cd ..

# Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## Step 3 — Clone your project repo (Cell 2)

```python
# Cell 2 — Get your project code
!git clone https://github.com/raidx545/interior-ai.git
%cd interior-ai

# Add project root to Python path
import sys
sys.path.insert(0, '/content/interior-ai')

print("✅ Project loaded")
```

> **If you haven't pushed to GitHub yet**, upload `vision_agent.py` and `contracts.py` manually using the Colab file browser (📁 icon on left), then run:
> ```python
> import sys; sys.path.insert(0, '/content')
> ```

---

## Step 4 — Run the Vision Agent (Cell 3)

```python
# Cell 3 — Run Vision Agent end-to-end
from contracts import AgentInput
from agents.vision_agent import run

test_input = AgentInput(
    image_url="https://images.unsplash.com/photo-1618220179428-22790b461013?w=1080",
    user_intent="Make it Scandinavian and cozy",
    budget_inr=45000,
)

result = run(test_input)
print(result.model_dump_json(indent=2))
```

**Expected output:**
```json
{
  "depth_map_path": "/tmp/interior_ai/depth/depth.png",
  "objects": [
    {"label": "couch", "confidence": 0.94, "bbox": [120, 200, 480, 420]},
    ...
  ],
  "room_dims_estimate": {"w_m": 4.2, "h_m": 2.8, "d_m": 3.7}
}
```

---

## Step 5 — Visualise the depth map (Cell 4)

```python
# Cell 4 — Display depth map
from PIL import Image
import matplotlib.pyplot as plt

depth_img = Image.open(result.depth_map_path)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(Image.open(
    # download original for comparison
    __import__('io').BytesIO(__import__('urllib.request', fromlist=['urlopen']).request.urlopen(test_input.image_url).read())
))
axes[0].set_title("Original Room")
axes[0].axis("off")

axes[1].imshow(depth_img, cmap="inferno")
axes[1].set_title("ZoeDepth — Depth Map (darker = closer)")
axes[1].axis("off")

plt.tight_layout()
plt.show()
print(f"Detected objects: {[o.label for o in result.objects]}")
print(f"Room dims: {result.room_dims_estimate}")
```

---

## Step 6 — Expose as API via ngrok (Cell 5)
*Only needed when you want your Mac FastAPI to call the Vision Agent*

```python
# Cell 5 — Serve Vision Agent as a FastAPI endpoint
!pip install fastapi uvicorn pyngrok nest-asyncio -q

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI
from pyngrok import ngrok
import uvicorn, threading, json

app = FastAPI()

@app.post("/vision")
def vision_endpoint(payload: dict):
    from contracts import AgentInput
    from agents.vision_agent import run

    agent_input = AgentInput(**payload)
    result = run(agent_input)

    # Return as JSON (depth_map_path is local to Colab, upload to Cloudinary here)
    return result.model_dump()

# Start tunnel
public_url = ngrok.connect(8000)
print(f"✅ Vision Agent API: {public_url}/vision")
print("→ Set GPU_VISION_URL={public_url} in your Mac .env")

# Run server
threading.Thread(
    target=uvicorn.run,
    args=(app,),
    kwargs={"host": "0.0.0.0", "port": 8000}
).start()
```

---

## Important Notes

| Thing | Detail |
|---|---|
| **First run time** | ~5-10 min (downloads YOLOv8 weights, SAM checkpoint ~2.4 GB, ZoeDepth weights) |
| **After first run** | Models cached, subsequent runs ~25 sec |
| **SAM checkpoint** | Auto-downloaded to `/tmp/sam_vit_h_4b8939.pth` (2.4 GB) |
| **Depth output** | Saved to `/tmp/interior_ai/depth/depth.png` |
| **T4 VRAM budget** | ZoeDepth ~2 GB + YOLOv8 ~1 GB + SAM ~4 GB ≈ 7 GB / 15 GB total |
| **ngrok session** | Free tier: 2 hours. Restart cell to renew. |
