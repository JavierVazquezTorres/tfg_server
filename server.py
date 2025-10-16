import os, tempfile, uuid, logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import numpy as np

from transcribe import transcribe_to_json_fast

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tfg-server")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

            # LOG TEMPRANO: ya sabemos que el POST llegó
            size_kb = len(raw) / 1024.0
            try:
                y, sr = sf.read(inpath, dtype="float32", always_2d=False)
                if y.ndim == 2:
                    y = np.mean(y, axis=1)
                dur = len(y) / float(sr)
            except Exception:
                y, sr, dur = None, None, None

            log.info("[%s] recibido name=%s size=%.0fKB sr=%s dur=%ss",
                     tag, file.filename, size_kb, sr, f"{dur:.2f}" if dur else "?")

            # Procesa (internamente recortamos a máx. 15 s para ir rápido)
            result = transcribe_to_json_fast(inpath)

        # LOG de verificación: primeras notas
        sample = result.get("notes", [])[:3]
        log.info("[%s] tempo=%s notas=%d sample=%s",
                 tag, result.get("tempo"), len(result.get("notes", [])), sample)
        return JSONResponse(result)

    except Exception as e:
        log.exception("[%s] error en /transcribe", tag)
        return PlainTextResponse(str(e), status_code=500)
