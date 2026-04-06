from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import asyncio
import os

app = FastAPI(title="Echo Detection WebGL Visualizer")

# Serve static files
app.mount("/static", StaticFiles(directory="/app/echo-detection-system/web/static"), name="static")

# Load nanograph data
with open("/app/echo-detection-system/outputs/nanograph.json") as f:
    NANOGRAPH_DATA = json.load(f)

@app.get("/")
async def root():
    with open("/app/echo-detection-system/web/index.html") as f:
        return HTMLResponse(f.read())

@app.get("/api/data")
async def get_data():
    return NANOGRAPH_DATA

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Stream nanograph data in chunks
        nodes = NANOGRAPH_DATA['nodes']
        batch_size = 10
        
        for i in range(0, len(nodes), batch_size):
            batch = {
                'type': 'node_batch',
                'data': nodes[i:i+batch_size],
                'progress': min(100, int(100 * (i + batch_size) / len(nodes)))
            }
            await websocket.send_json(batch)
            await asyncio.sleep(0.1)  # Simulate real-time streaming
        
        # Send completion
        await websocket.send_json({'type': 'complete', 'message': 'Stream complete'})
        
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
