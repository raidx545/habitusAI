# Interior AI — Agentic Interior Design Assistant

> Transform any room photo into a fully designed, shoppable interior using a 4-agent AI pipeline powered by Claude, SDXL, and ControlNet.

---

## Problem Statement

Professional interior design is inaccessible to most people — expensive, slow, and requires expertise that the average homeowner simply does not have. Users struggle to visualize how a space could look, which furniture fits their room dimensions, and how to make design decisions without expert guidance.

Traditional design tools require technical skill. Generic AI image generators produce beautiful rooms that look nothing like the user's actual space. There is no tool that takes *your* room and makes it *better* — while keeping its walls, windows, and structure intact.

---

## What We Built

**Interior AI** is an agentic system that takes a single room photograph and a plain-English design request and returns a photorealistic redesign of that exact room — with a floor plan, furniture placement guidance, and a real IKEA shopping list with pricing.

The system uses four specialized AI agents running in parallel, orchestrated by Claude, with SDXL + ControlNet handling the final image generation. The depth map extracted from the original photo is used as a ControlNet conditioning image — ensuring the room's architectural structure (walls, windows, doors, floor) is preserved while only the interior design changes.

---

## Architecture

```
User photo + design intent
        │
        ▼
Claude (Orchestrator)
Parses intent → dispatches 4 agents in parallel
        │
   ┌────┼────┬──────────┐
   ▼    ▼    ▼          ▼
Vision  Style  Layout  Commerce
Agent   Agent  Agent   Agent
   └────┴────┴──────────┘
        │
        ▼
Claude (Aggregator)
Merges all outputs → structured SDXL prompt
        │
        ▼
ControlNet (depth map) + SDXL Base (30 steps)
        │
        ▼
SDXL Refiner (img2img sharpening)
        │
        ▼
Redesigned room + Floor plan + IKEA shopping list
```

---

## The Four Agents

### Agent 1 — Vision Agent
**Owner:** Raj Porwal

Extracts spatial understanding from the room image.

- **ZoeDepth** — metric monocular depth estimation. Produces a grayscale depth map that captures the 3D spatial layout of the room. This map is fed directly to ControlNet as the structure anchor.
- **YOLOv8** — detects existing furniture and architectural features (windows, doors, walls) with bounding boxes and confidence scores.
- **Segment Anything (SAM)** — generates precise segmentation masks for each detected region.

**Output:** `depth.png` + object bounding box JSON + estimated room dimensions

---

### Agent 2 — Style Agent
**Owner:** Raj Porwal

Understands and classifies the visual style of the room and the user's intent.

- **CLIP ViT-L/14** — encodes the room image into a 512-dimensional style embedding vector.
- **ChromaDB** — performs similarity search against a curated vector database of labeled interior design styles (minimalist, maximalist, Scandinavian, Japandi, industrial3, etc.).
- Maps the user's text intent ("make it cozy", "Scandinavian vibes") to precise design language that SDXL responds to well.

**Output:** `style_label` + `style_keywords` string for SDXL prompt injection

---

### Agent 3 — Layout Agent
**Owner:** Friend

Generates a floor plan and determines where furniture should be placed given the room's actual geometry.

- **HorizonNet** — estimates room layout boundaries and wall positions from a single image.
- **RoomFormer** — generates a structured floor plan representation.
- **Shapely** — applies spatial placement rules (clearance distances, Vastu guidelines, traffic flow) to determine valid furniture zones.

**Output:** Floor plan image + furniture placement zones JSON + clearance validation

---

### Agent 4 — Commerce Agent
**Owner:** Friend

Finds real, purchasable furniture that matches the design recommendations within the user's budget.

- **IKEA Connectivity API** — searches IKEA's product catalog by category and style keywords.
- **Redis cache** — caches product results to avoid redundant API calls during a session.
- Filters by user-defined budget and returns SKUs, prices, and direct product URLs.
- Falls back to SerpAPI product search if IKEA API is unavailable.

**Output:** Ranked product list with names, prices, images, and purchase links

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestrator | Claude 3.5 Sonnet (Anthropic API) |
| Agent framework | LangGraph + Celery |
| Depth estimation | ZoeDepth |
| Object detection | YOLOv8 |
| Style embedding | CLIP ViT-L/14 |
| Vector search | ChromaDB |
| Room layout | HorizonNet + RoomFormer |
| Spatial rules | Shapely |
| Product search | IKEA API + SerpAPI fallback |
| Cache | Redis |
| Image generation | Stable Diffusion XL Base 1.0 |
| Structure control | ControlNet Depth |
| Image refinement | SDXL Refiner (img2img) |
| Diffusion library | HuggingFace Diffusers |
| Backend | FastAPI (Python, async) |
| Task queue | Celery + Redis broker |
| Database | PostgreSQL |
| Image storage | Cloudinary + AWS S3 |
| Frontend | Next.js + TailwindCSS |
| Real-time updates | Socket.IO |
| 3D viewer | Three.js + Open3D |
| Before/after UI | React Compare Slider |
| GPU runtime | CUDA + PyTorch |
| Containerization | Docker + Docker Compose |
| Frontend deploy | Vercel |
| CI/CD | GitHub Actions |

---

## Key Innovation

Most AI image tools generate a *new* room inspired by your photo. We generate a redesign of *your actual room*.

The critical insight is the **depth map as ControlNet conditioning signal**. ZoeDepth converts the room photo into a precise grayscale spatial map. This map is fed into ControlNet at every one of SDXL's 30 denoising steps — applying a structural correction force that prevents the model from drifting away from the original room's geometry. The sliding glass door stays on the left. The arched doorway stays on the right. The hardwood floor stays at the bottom. Only the interior design changes.

Claude acts as the reasoning layer that bridges the gap between a user saying "make it maximalist and colorful" and the precise, structured prompt that SDXL needs to produce photographic-quality output.

---

## Novel Features

- **Structure-preserving generation** — ControlNet depth conditioning keeps architectural elements intact
- **Parallel 4-agent pipeline** — all agents run simultaneously via Celery, total pipeline time ~25 seconds
- **Real shoppable output** — every item placed in the generated room has a corresponding IKEA product link and price
- **Vastu compliance layer** — furniture placement zones are validated against Vastu Shastra rules (India-specific)
- **Live agent progress** — Socket.IO streams real-time status of each agent to the frontend
- **Before/after comparison** — interactive drag slider to compare original and redesigned room
- **3D point cloud viewer** — Open3D + Three.js renders a navigable 3D version of the depth-reconstructed room

---

## Team

| Member | Role | Agents |
|---|---|---|
| Raj Porwal | Backend + AI Lead | Vision Agent, Style Agent, Claude Orchestrator |
| [Friend] | Backend + Integration | Layout Agent, Commerce Agent, SDXL Pipeline |

---

## Repository Structure

```
interior-ai/
├── contracts.py              # Shared Pydantic input/output schemas
├── mocks.py                  # Mock agent outputs for parallel development
├── agents/
│   ├── vision_agent.py       # ZoeDepth + YOLOv8
│   ├── style_agent.py        # CLIP + ChromaDB
│   ├── layout_agent.py       # HorizonNet + Shapely
│   └── commerce_agent.py     # IKEA API + fallback
├── orchestrator/
│   └── aggregator.py         # Claude aggregation + SDXL prompt builder
├── generation/
│   └── sdxl_pipeline.py      # ControlNet + SDXL + Refiner
├── api/
│   └── main.py               # FastAPI endpoints + Celery tasks
├── frontend/                 # Next.js application
├── docker-compose.yml
└── requirements.txt
```

---

## Running Locally

```bash
# Clone and install
git clone https://github.com/raidx545/interior-ai
cd interior-ai
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add: ANTHROPIC_API_KEY, CLOUDINARY_URL, DATABASE_URL

# Start all services
docker-compose up

# Run integration test
python integration_test.py
```

**GPU required for SDXL inference.** Use Google Colab T4 (free) for hackathon demo:
```bash
# Expose Colab backend via ngrok
ngrok http 8000
# Update NEXT_PUBLIC_API_URL in frontend/.env.local
```

---

## Hackathon

Built at **RIFT Noida 24-Hour Hackathon** — March 2026

*Agentic AI track — Interior design use case*
