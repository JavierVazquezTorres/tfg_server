# server.py (añade estos imports arriba)
import io, os, tempfile, uuid, logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
import librosa

from transcribe import transcribe_to_json           # tu DSP actual
from transcribe_ml import segment_pitch_track, hz_to_note_name

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

# --------- IA MONOFÓNICA (CREPE) ----------
@app.post("/transcribe_crepe")
async def transcribe_crepe(file: UploadFile = File(...)):
    tag = uuid.uuid4().hex[:6]
    try:
        import crepe  # carga bajo demanda
        raw = await file.read()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, file.filename or f"{tag}.wav")
            with open(path, "wb") as f:
                f.write(raw)

            # lee audio; re-muestrea a 16000 Hz recomendado por CREPE
            y, sr = librosa.load(path, sr=16000, mono=True)
            # CREPE espera float32 en [-1,1]
            y = y.astype(np.float32, copy=False)

            # viterbi=True mejora continuidad; step_size ms (10 ≈ 100 Hz)
            time, frequency, confidence, activation = crepe.predict(
                y, sr, viterbi=True, step_size=10, model_capacity="tiny"
            )
            # segmenta en notas con umbral de confianza simple
            f0 = np.where(confidence >= 0.3, frequency, np.nan)
            notes = segment_pitch_track(time, f0, min_dur=0.08)

            # tempo opcional con librosa
            try:
                tempo = float(librosa.beat.tempo(y=y, sr=sr, hop_length=256)[0])
            except Exception:
                tempo = None

            res = {"tempo": tempo, "notes": notes}
            log.info("[%s] crepe notes=%d", tag, len(notes))
            return JSONResponse(res)
    except Exception as e:
        log.exception("[%s] error crepe", tag)
        return PlainTextResponse(str(e), status_code=500)

# --------- IA POLIFÓNICA (BASIC PITCH) ----------
@app.post("/transcribe_poly")
async def transcribe_poly(file: UploadFile = File(...)):
    tag = uuid.uuid4().hex[:6]
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict
        raw = await file.read()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, file.filename or f"{tag}.wav")
            with open(path, "wb") as f:
                f.write(raw)

            # Basic Pitch devuelve (note_events, note_times, note_frames)
            # note_events: [ (start_time, end_time, pitch_midi, amplitude), ... ]
            note_events, _, _ = predict(
                audio_path_list=[path],
                model_or_model_path=ICASSP_2022_MODEL_PATH
            )

            events = note_events[0]  # lista del primer archivo
            notes = []
            for (t0, t1, midi, amp) in events:
                # convierte MIDI a nota tipo "C#4"
                hz = librosa.midi_to_hz(midi)
                notes.append({
                    "pitch": hz_to_note_name(hz),
                    "start": float(t0),
                    "end": float(t1),
                    "vel": float(amp)
                })

            # tempo aproximado
            y, sr = librosa.load(path, sr=16000, mono=True)
            try:
                tempo = float(librosa.beat.tempo(y=y, sr=sr, hop_length=256)[0])
            except Exception:
                tempo = None

            res = {"tempo": tempo, "notes": notes}
            log.info("[%s] basic-pitch notes=%d", tag, len(notes))
            return JSONResponse(res)
    except Exception as e:
        log.exception("[%s] error basic-pitch", tag)
        return PlainTextResponse(str(e), status_code=500)

# --------- Tu endpoint actual (DSP clásico) sigue disponible ----------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tag = uuid.uuid4().hex[:6]
    try:
        with tempfile.TemporaryDirectory() as td:
            inpath = os.path.join(td, f"{tag}_{file.filename or 'audio.wav'}")
            raw = await file.read()
            with open(inpath, "wb") as f:
                f.write(raw)
            result = transcribe_to_json(inpath)
        return JSONResponse(result)
    except Exception as e:
        log.exception("[%s] error", tag)
        return PlainTextResponse(str(e), status_code=500)
