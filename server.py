import os, tempfile, uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from transcribe import transcribe_to_json

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory() as td:
        inpath = os.path.join(td, f"{uid}_{file.filename}")
        with open(inpath, "wb") as f:
            f.write(await file.read())
        # procesa y devuelve JSON con tempo/compás/notas
        result = transcribe_to_json(inpath)
        return JSONResponse(result)

@app.get("/")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # solo para local: Render usará CMD del Dockerfile
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
