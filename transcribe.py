import numpy as np
import librosa

SR = 16000
HOP = 512            # 16kHz/512 ≈ 31.25 fps → ~32 ms por frame (más rápido)
FRAME = 2048
MIN_DUR = 0.05       # 50 ms
MAX_SECONDS = 15.0   # recorta a 15 s para que nunca se eternice

def hz_to_note_name(hz: float) -> str:
    if not np.isfinite(hz) or hz <= 0:
        return "Rest"
    return librosa.hz_to_note(hz, cents=False)

def _segments_from_voicing(voiced_flag: np.ndarray):
    v = np.asarray(voiced_flag, dtype=np.int8)
    if v.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.pad(v, (1, 1), mode="constant")))
    out = []
    for i in range(0, len(edges), 2):
        start = edges[i]
        end = edges[i + 1] if i + 1 < len(edges) else len(v)
        if end > start:
            out.append((start, end))
    return out

def transcribe_to_json_fast(path: str) -> dict:
    # Carga mono a SR y recorta a MAX_SECONDS
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) > int(MAX_SECONDS * sr):
        y = y[: int(MAX_SECONDS * sr)]

    # Detección de pitch monofónica (pYIN)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=FRAME,
        hop_length=HOP,
        center=False,  # frame t corresponde a t*HOP exactamente
    )

    if voiced_flag is None:
        voiced_flag = np.isfinite(f0)

    segments = _segments_from_voicing(voiced_flag)

    notes = []
    for s_f, e_f in segments:
        start_sample = int(s_f * HOP)
        end_sample = int(min(len(y), e_f * HOP))
        start_t = start_sample / sr
        end_t = end_sample / sr
        dur = end_t - start_t
        if dur < MIN_DUR:
            continue

        f0_seg = f0[s_f:e_f]
        f0_valid = f0_seg[np.isfinite(f0_seg)]
        if f0_valid.size == 0:
            continue

        hz_med = float(np.median(f0_valid))
        notes.append({
            "pitch": hz_to_note_name(hz_med),
            "start": float(start_t),
            "end": float(end_t),
        })

    try:
        tempo = float(librosa.beat.tempo(y=y, sr=sr, hop_length=HOP)[0])
    except Exception:
        tempo = None

    return {"tempo": tempo, "notes": notes}
