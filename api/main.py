"""
api/main.py — FastAPI Backend
Exposes the HabitusAI Orchestrator + Aggregator as a generic REST API.

Run the server locally:
$ pip install fastapi uvicorn pydantic
$ uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Add project root so imports work
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from contracts import AgentInput
from orchestrator.orchestrator import orchestrate
from orchestrator.aggregator import aggregate

app = FastAPI(
    title="HabitusAI Backend",
    description="API for the 4-Agent Interior Design Orchestrator.",
    version="1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g. localhost:3000, Colab, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ── API Models ────────────────────────────────────────────────────────────────

class DesignRequest(BaseModel):
    image_url: str
    user_intent: str
    budget_inr: Optional[int] = 50000

class DesignResponse(BaseModel):
    sdxl_prompt: str
    negative_prompt: str
    design_summary: str
    depth_map_path: str
    floor_plan_path: str
    products: list


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "HabitusAI API is running!"}


@app.post("/design", response_model=DesignResponse)
async def generate_design(req: DesignRequest):
    """
    Main endpoint. Synchronously runs the Orchestrator (4 agents in parallel)
    and then runs the Aggregator to generate the final SDXL prompt + products.
    """
    try:
        agent_input = AgentInput(
            image_url=req.image_url,
            user_intent=req.user_intent,
            budget_inr=req.budget_inr
        )

        # 1. Orchestrate (runs Groq parser + 4 agents concurrently)
        print(f"[API] Starting orchestration for image: {req.image_url}")
        orchestrator_output = await orchestrate(agent_input)

        # 2. Aggregate (synthesize findings into an SDXL prompt)
        print("[API] Aggregating results...")
        final_output = aggregate(orchestrator_output)

        return final_output

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks/design")
async def generate_design_celery(req: DesignRequest, background_tasks: BackgroundTasks):
    """
    Async/Celery Placeholder Endpoint.
    Instead of making the frontend wait 60s for all agents,
    we'd return a task_id and run the orchestration in the background using Celery.
    """
    # In a full production setup with Redis/Celery:
    # task = design_task.delay(req.dict())
    # return {"task_id": task.id}
    
    return {
        "message": "Async task enqueued successfully. Use WebSockets to listen for updates.",
        "task_id": "dummy-celery-task-id-1234"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
