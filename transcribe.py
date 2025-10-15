import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000
HOP_LENGTH = 160            # 10 ms a 16 kHz
FRAME_LENGTH = 2048
FMIN_HZ, FMAX_HZ = 50.0, 1000.0
MIN_NOTE_DUR = 0.06         # 60 ms
DEFAULT_BPM, DEFAULT_TS = 100, "4/4"

def _load_audio(path, sr=TARGET_SR, max_sec=30):
    y, in_sr = sf.read(path, always_2d=False)
    if y.ndim > 1: y = np.mean(y, axis=1)
    if in_sr != sr: y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
    if max_sec and len(y) > sr * max_sec: y = y[: sr * max_sec]
    y = librosa.util.normalize(y).astype(np.float32)
    return y, sr

def _f0_librosa_pyin(y, sr, fmin=FMIN_HZ, fmax=FMAX_HZ, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    # pyin devuelve f0 (Hz), voiced_flag y probabilidad
    f0, vflag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length
    )
    # f0 es array con NaN cuando no hay voz
    f0 = np.where(vflag, np.nan_to_num(f0, nan=0.0), 0.0).astype(np.float32)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    return times, f0

def _segment_notes(times, f0_hz, min_dur=MIN_NOTE_DUR):
    notes, s, voiced = [], None, (f0_hz > 0)
    n = len(f0_hz)
    for i in range(n):
        last = (i == n - 1)
        if voiced[i] and s is None: s = i
        if ((not voiced[i]) or last) and s is not None:
            e = i if not voiced[i] else (i + 1)
            dur = max(0.0, times[e-1] - times[s]) if (e-1) > s else 0.0
            if dur >= min_dur:
                hz_med = float(np.median(f0_hz[s:e][f0_hz[s:e] > 0]))
                notes.append((times[s], times[e-1], hz_med))
            s = None
    return notes

def _hz_to_midi(hz): return 69 + 12 * np.log2(hz / 440.0)

def _midi_to_name(m):
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    m = int(round(m)); return f"{names[m % 12]}{(m // 12) - 1}"

def _quantize_len(sec, bpm=DEFAULT_BPM):
    beat = 60.0 / bpm
    grid = [("whole", 4*beat), ("half", 2*beat), ("quarter", beat),
            ("eighth", beat/2), ("16th", beat/4)]
    name, q = min(grid, key=lambda x: abs(sec - x[1]))
    return name, q

def transcribe_to_json(path, bpm=DEFAULT_BPM, time_sig=DEFAULT_TS):
    y, sr = _load_audio(path)
    times, f0 = _f0_librosa_pyin(y, sr)
    seg = _segment_notes(times, f0)

    out = []
    for t0, t1, hz in seg:
        midi = _hz_to_midi(hz)
        pname = _midi_to_name(midi)
        dur = t1 - t0
        fig, _ = _quantize_len(dur, bpm)
        out.append({"pitch": pname, "duration": fig})

    return {"tempo": bpm, "time_signature": time_sig, "notes": out}
