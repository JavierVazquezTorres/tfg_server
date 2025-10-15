import os, tempfile, uuid, logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transcribe import transcribe_to_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tfg-server")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        uid = uuid.uuid4().hex[:8]
        logger.info(f"Recv file: {file.filename} ({file.content_type}) uid={uid}")
        with tempfile.TemporaryDirectory() as td:
            inpath = os.path.join(td, f"{uid}_{file.filename}")
            with open(inpath, "wb") as f:
                f.write(await file.read())
            logger.info(f"Saved to {inpath}, size={os.path.getsize(inpath)} bytes")
            result = transcribe_to_json(inpath)
            logger.info(f"OK uid={uid} notes={len(result.get('notes', []))}")
            return JSONResponse(result)
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # Solo para local. En Render usa el CMD del Dockerfile con $PORT
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
