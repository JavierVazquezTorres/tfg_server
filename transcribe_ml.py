# transcribe_ml.py
import numpy as np
import librosa

A4 = 440.0

def hz_to_note_name(hz: float) -> str:
    if not np.isfinite(hz) or hz <= 0:
        return "Rest"
    return librosa.hz_to_note(hz, cents=False)

def segment_pitch_track(times: np.ndarray, f0_hz: np.ndarray, min_dur=0.08):
    """
    Segmenta una curva de pitch en notas con start/end (segundos).
    f0_hz puede traer NaN en frames sin voz.
    """
    notes = []
    voiced = np.isfinite(f0_hz) & (f0_hz > 0)
    if times.size == 0:
        return notes

    i = 0
    N = len(times)
    while i < N:
        if not voiced[i]:
            i += 1
            continue
        start = i
        freqs = []
        while i < N and voiced[i]:
            freqs.append(f0_hz[i])
            i += 1
        end = i  # exclusivo
        t0 = float(times[start])
        t1 = float(times[min(end - 1, N - 1)])
        if (t1 - t0) < min_dur or len(freqs) == 0:
            continue
        hz_med = float(np.median(freqs))
        notes.append({"pitch": hz_to_note_name(hz_med), "start": t0, "end": t1})
    return notes
