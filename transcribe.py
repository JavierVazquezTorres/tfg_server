import numpy as np
import librosa

HOP = 256          # igual que en la app: 16 kHz → 256 → 16 ms por frame
FRAME = 2048       # ventana de análisis
SR = 16000         # la app graba a 16 kHz
MIN_DUR = 0.05     # descarta segmentos de menos de 50 ms


def hz_to_note_name(hz: float) -> str:
    """Convierte frecuencia a nombre de nota (C4, A#3...)."""
    if not np.isfinite(hz) or hz <= 0:
        return "Rest"
    return librosa.hz_to_note(hz, cents=False)


def _segments_from_voicing(voiced_flag: np.ndarray) -> list[tuple[int, int]]:
    """
    Devuelve pares (inicio_frame, fin_frame_exclusivo) para cada tramo con voz.
    No usa tiempos de librosa; lo calculamos a partir de frames -> muestras.
    """
    v = np.asarray(voiced_flag, dtype=np.int8)
    # Asegura que no es todo None/False
    if v.size == 0:
        return []
    # bordes de cambio 0→1 o 1→0 (incluimos pads para capturar extremos)
    edges = np.flatnonzero(np.diff(np.pad(v, (1, 1), mode="constant")))
    # si empezamos en estado "1" (con voz), el primer segmento empieza en 0
    segments = []
    for i in range(0, len(edges), 2):
        start = edges[i]
        end = edges[i + 1] if i + 1 < len(edges) else len(v)
        # guardamos [start, end) en frames
        if end > start:
            segments.append((start, end))
    return segments


def transcribe_to_json(path: str) -> dict:
    # Carga mono a 16 kHz
    y, sr = librosa.load(path, sr=SR, mono=True)

    # Estimación de tono por frame con pYIN (monofónico)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=FRAME,
        hop_length=HOP,
        center=False,  # << importante para que el frame t corresponda a t*HOP
    )

    # Si pYIN no devuelve flags, deducimos "voz" como frames con f0 finito
    if voiced_flag is None:
        voiced_flag = np.isfinite(f0)

    # Segmenta tramos de voz por frames contiguos
    segments = _segments_from_voicing(voiced_flag)

    notes = []
    for start_f, end_f in segments:
        # Convertimos frames a muestras y luego a segundos
        start_sample = int(start_f * HOP)
        end_sample = int(min(len(y), end_f * HOP))  # exclusivo

        start_t = start_sample / sr
        end_t = end_sample / sr
        dur = end_t - start_t
        if dur < MIN_DUR:
            continue

        # f0 del tramo (solo valores finitos)
        f0_seg = f0[start_f:end_f]
        f0_valid = f0_seg[np.isfinite(f0_seg)]
        if f0_valid.size == 0:
            continue

        hz_med = float(np.median(f0_valid))
        note_name = hz_to_note_name(hz_med)

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
