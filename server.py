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
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tag = uuid.uuid4().hex[:6]
    try:
        with tempfile.TemporaryDirectory() as td:
            inpath = os.path.join(td, file.filename or f"{tag}.wav")
            raw = await file.read()
            with open(inpath, "wb") as f:
                f.write(raw)
            res = transcribe_to_json(inpath)

        # Log: primeras 3 notas con tiempos
        sample = res.get("notes", [])[:3]
        log.info("[%s] tempo=%s sample=%s", tag, res.get("tempo"), sample)
        return JSONResponse(res)
    except Exception as e:
        log.exception("[%s] error en transcribe", tag)
        return PlainTextResponse(str(e), status_code=500)
