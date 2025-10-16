import numpy as np
import soundfile as sf
import librosa

# Parámetros
TARGET_SR = 8000       # trabajamos a 8 kHz (más rápido y es lo que graba la app)
HOP = 128              # 8k/128 = 62.5 fps -> ~16 ms por frame
FRAME = 2048
FMIN_HZ, FMAX_HZ = 60.0, 1200.0   # rango amplio (voz e instrumentos)
MIN_DUR = 0.05         # descarta notas < 50 ms
MAX_SEC = 20.0         # límite duro para no eternizar

def _load(path, sr=TARGET_SR, max_sec=MAX_SEC):
    y, in_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    if in_sr != sr:
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
    if max_sec and len(y) > int(sr * max_sec):
        y = y[: int(sr * max_sec)]
    return y, sr

def _segments_from_flags(flags: np.ndarray):
    v = np.asarray(flags, dtype=np.int8)
    if v.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.pad(v, (1, 1), mode="constant")))
    segs = []
    for i in range(0, len(edges), 2):
        s = edges[i]
        e = edges[i + 1] if i + 1 < len(edges) else len(v)
        if e > s:
            segs.append((s, e))
    return segs

def transcribe_to_json(path: str) -> dict:
    # Cargar y normalizar
    y, sr = _load(path)

    # Pitch por frame con pYIN (monofónico)
    f0, vflag, _ = librosa.pyin(
        y,
        fmin=FMIN_HZ, fmax=FMAX_HZ,
        sr=sr, frame_length=FRAME, hop_length=HOP,
        center=False
    )
    if vflag is None:
        vflag = np.isfinite(f0)

    # Segmentación por tramos de voz
    segs = _segments_from_flags(vflag)

    notes = []
    for s_f, e_f in segs:
        start_samp = int(s_f * HOP)
        end_samp   = int(min(len(y), e_f * HOP))
        start_t = start_samp / sr
        end_t   = end_samp / sr
        dur = end_t - start_t
        if dur < MIN_DUR:
            continue

        f0_seg = f0[s_f:e_f]
        f0_valid = f0_seg[np.isfinite(f0_seg)]
        if f0_valid.size == 0:
            continue

        hz_med = float(np.median(f0_valid))
        note_name = librosa.hz_to_note(hz_med, cents=False)

        notes.append({
            "pitch": note_name,
            "start": float(start_t),
            "end": float(end_t),
        })

    # Tempo aproximado (opcional)
    try:
        tempo = float(librosa.beat.tempo(y=y, sr=sr, hop_length=HOP)[0])
    except Exception:
        tempo = None

    return {"tempo": tempo, "notes": notes}
