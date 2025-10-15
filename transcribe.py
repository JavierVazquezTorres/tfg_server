import os
import numpy as np
import soundfile as sf
import librosa
import aubio

# Config
TARGET_SR = 16000
HOP_LENGTH = 160              # 10 ms a 16 kHz
FMIN = 50.0
FMAX = 1000.0
CONF_THRESH = 0.7             # confianza mínima YIN
MIN_NOTE_DUR = 0.06           # 60 ms
DEFAULT_BPM = 100
DEFAULT_TS = "4/4"

def _load_audio(path, sr=TARGET_SR, max_sec=30):
    # Lee con soundfile, re-muestrea con librosa (ligero comparado con torch)
    y, in_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:  # a mono
        y = np.mean(y, axis=1)
    if in_sr != sr:
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
    if max_sec and len(y) > sr * max_sec:
        y = y[: sr * max_sec]
    y = librosa.util.normalize(y)
    return y.astype(np.float32), sr

def _f0_aubio(y, sr, hop_length=HOP_LENGTH):
    # aubio.pitch(YIN) entrega f0 y confianza por frame
    win_s = 4 * hop_length
    pitch_o = aubio.pitch("yin", win_s, hop_length, sr)
    pitch_o.set_unit("Hz")
    pitch_o.set_tolerance(0.2)

    n_frames = 1 + (len(y) - 1) // hop_length
    f0 = np.zeros(n_frames, dtype=np.float32)
    conf = np.zeros(n_frames, dtype=np.float32)

    idx = 0
    for i in range(0, len(y) - hop_length + 1, hop_length):
        frame = y[i:i + hop_length]
        hz = float(pitch_o(frame)[0])
        c = float(pitch_o.get_confidence())
        f0[idx] = hz if (hz >= FMIN and hz <= FMAX) else 0.0
        conf[idx] = c
        idx += 1

    # Tiempos por frame
    times = np.arange(len(f0)) * (hop_length / sr)
    # Enmascara frames de baja confianza
    f0[conf < CONF_THRESH] = 0.0
    return times, f0

def _segment_notes(times, f0_hz, min_dur=MIN_NOTE_DUR):
    notes, start = [], None
    voiced = f0_hz > 0
    n = len(f0_hz)
    for i in range(n):
        last = (i == n - 1)
        if voiced[i] and start is None:
            start = i
        if ((not voiced[i]) or last) and start is not None:
            end = i if not voiced[i] else (i + 1)
            dur = max(0.0, times[end - 1] - times[start]) if (end - 1) > start else 0.0
            if dur >= min_dur:
                hz_slice = f0_hz[start:end]
                hz_med = np.median(hz_slice[hz_slice > 0])
                notes.append((times[start], times[end - 1], float(hz_med)))
            start = None
    return notes

def _hz_to_midi(hz):
    return 69 + 12 * np.log2(hz / 440.0)

def _midi_to_name(m):
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    m = int(round(m))
    return f"{names[m % 12]}{(m // 12) - 1}"

def _quantize_len(sec, bpm=DEFAULT_BPM):
    beat = 60.0 / bpm
    grid = [("whole", 4*beat), ("half", 2*beat), ("quarter", beat),
            ("eighth", beat/2), ("16th", beat/4)]
    name, q = min(grid, key=lambda x: abs(sec - x[1]))
    return name, q

def transcribe_to_json(path, bpm=DEFAULT_BPM, time_sig=DEFAULT_TS):
    y, sr = _load_audio(path)
    times, f0 = _f0_aubio(y, sr)
    seg = _segment_notes(times, f0)

    notes_json = []
    for t0, t1, hz in seg:
        midi = _hz_to_midi(hz)
        pname = _midi_to_name(midi)
        dur = t1 - t0
        fig, _ = _quantize_len(dur, bpm=bpm)
        notes_json.append({"pitch": pname, "duration": fig})

    return {"tempo": bpm, "time_signature": time_sig, "notes": notes_json}
