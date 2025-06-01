from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.graph import graph
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
def run_graph(arxiv_paper_url: str):
    try:
        result = graph.invoke({"arxiv_paper_url": arxiv_paper_url})
        # Convert the result to a JSON-serializable format
        if hasattr(result, 'dict'):  # If it's a Pydantic model
            result = result.dict()
        elif hasattr(result, 'model_dump'):  # For newer Pydantic versions
            result = result.model_dump()
        return result  # FastAPI will handle JSON serialization
    except Exception as e:
        return {"error": f"Failed to process request: {str(e)}"}


@app.get("/stream")
async def run_graph_stream(arxiv_paper_url: str):
    print("arxiv_paper_url as query param", arxiv_paper_url)
    async def event_generator():
        async for msg, metadata in graph.astream({"arxiv_paper_url": arxiv_paper_url}, stream_mode="messages"):
            try:
                # print(metadata)
                # Only stream serializable part of step
                yield f"data: {json.dumps({"content": msg.content, "langgraph_node": metadata.get("langgraph_node")})}\n\n"

            except TypeError as e:
                # Optional: log the error and skip
                print("Serialization error:", e)
                continue
            await asyncio.sleep(0.1)
        yield "event: done\ndata: stream complete\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
