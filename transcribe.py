import numpy as np
import librosa
import librosa.display

# Utilidad: convertir Hz a nombre de nota cercano (C4, A#3, etc.)
def hz_to_note_name(hz: float) -> str:
    if not np.isfinite(hz) or hz <= 0:
        return "Rest"
    # librosa.hz_to_note devuelve strings tipo 'C#4'
    return librosa.hz_to_note(hz, cents=False)

def transcribe_to_json(path: str) -> dict:
    # Carga mono a 16 kHz (como graba la app)
    y, sr = librosa.load(path, sr=16000, mono=True)

    # Pitch tracking con pYIN (monofónico). Ajusta fmin/fmax a tu rango
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=256,
        center=True,
    )
    # Tiempo por frame
    frames = np.arange(len(f0))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=256)

    # Estima tempo (opcional)
    try:
        tempo = float(librosa.beat.tempo(y=y, sr=sr, hop_length=256)[0])
    except Exception:
        tempo = None

    # Segmentación: agrupa frames "voiced" consecutivos en notas
    notes = []
    min_duration = 0.05  # 50 ms para evitar notas demasiado cortas
    i = 0
    N = len(f0)

    while i < N:
        if not voiced_flag[i]:
            i += 1
            continue

        # inicio del segmento con voz
        start_idx = i
        segment_freqs = []

        while i < N and voiced_flag[i]:
            if np.isfinite(f0[i]) and f0[i] > 0:
                segment_freqs.append(f0[i])
            i += 1
        end_idx = i  # excluyente

        start_t = float(times[start_idx])
        end_t = float(times[min(end_idx - 1, N - 1)])

        # Asegura que end_t > start_t
        if end_t <= start_t:
            continue

        duration = end_t - start_t
        if duration < min_duration:
            continue

        # Nota del segmento: mediana de f0 del tramo
        if len(segment_freqs) == 0:
            # Sin f0 válida; ignora
            continue

        hz_med = float(np.median(segment_freqs))
        note_name = hz_to_note_name(hz_med)

        notes.append({
            "pitch": note_name,
            "start": start_t,
            "end": end_t,
        })

    result = {
        "tempo": tempo,
        "notes": notes,
    }
    return result
