import os, tempfile, uuid, logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from transcribe import transcribe_to_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tfg-server")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health(): return {"status":"ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex[:8]
    try:
        logger.info(f"[{uid}] start name={file.filename} ctype={file.content_type}")
        with tempfile.TemporaryDirectory() as td:
            inpath = os.path.join(td, f"{uid}_{file.filename or 'audio.wav'}")
            raw = await file.read(); logger.info(f"[{uid}] bytes={len(raw)}")
            with open(inpath, "wb") as f: f.write(raw)
            res = transcribe_to_json(inpath)
            logger.info(f"[{uid}] ok notes={len(res.get('notes',[]))}")
            return JSONResponse(res)
    except Exception as e:
        logger.exception(f"[{uid}] error")
        return PlainTextResponse(str(e), status_code=500)
