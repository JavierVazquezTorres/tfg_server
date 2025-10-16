import os, tempfile, uuid, logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from transcribe import transcribe_to_json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tfg-server")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tag = uuid.uuid4().hex[:6]
    try:
        log.info("[%s] start name=%s ctype=%s", tag, file.filename, file.content_type)
        with tempfile.TemporaryDirectory() as td:
            inpath = os.path.join(td, f"{tag}_{file.filename or 'audio.wav'}")
            raw = await file.read()
            log.info("[%s] bytes=%d", tag, len(raw))
            with open(inpath, "wb") as f:
                f.write(raw)

            result = transcribe_to_json(inpath)
            sample = result.get("notes", [])[:3]
            log.info("[%s] tempo=%s notes=%d sample=%s",
                     tag, result.get("tempo"), len(result.get("notes", [])), sample)

        return JSONResponse(result)
    except Exception as e:
        log.exception("[%s] error", tag)
        return PlainTextResponse(str(e), status_code=500)
