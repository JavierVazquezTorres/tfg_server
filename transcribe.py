import numpy as np, librosa, torch, torchcrepe, soundfile as sf

def _load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = librosa.util.normalize(y)
    return y, sr

def _f0_torchcrepe(y, sr, hop_length=160, fmin=50.0, fmax=1000.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    f0, pd = torchcrepe.predict(
        x, sr, hop_length, fmin, fmax,
        model="full", device=device, batch_size=1024, return_periodicity=True
    )
    f0 = torchcrepe.filter.median(f0, 3).squeeze(0).cpu().numpy()
    pd = torchcrepe.filter.mean(pd, 3).squeeze(0).cpu().numpy()
    f0[pd < 0.5] = 0.0
    times = np.arange(len(f0)) * (hop_length / sr)
    return times, f0

def _segment_notes(times, f0_hz, min_dur=0.06):
    notes = []
    voiced = f0_hz > 0
    s = None
    for i in range(len(f0_hz)):
        if voiced[i] and s is None: s = i
        if (not voiced[i] or i == len(f0_hz)-1) and s is not None:
            e = i if not voiced[i] else i+1
            dur = times[e-1] - times[s] if e-1 > s else 0
            if dur >= min_dur:
                pitch = np.median(f0_hz[s:e][f0_hz[s:e] > 0])
                notes.append((times[s], times[e-1], pitch))
            s = None
    return notes

def _hz_to_midi(hz): return 69 + 12*np.log2(hz/440.0)

def _quantize_len(sec, bpm=100.0):
    beat = 60.0 / bpm
    grid = [("whole",4*beat),("half",2*beat),("quarter",beat),
            ("eighth",beat/2),("16th",beat/4)]
    name, _ = min(grid, key=lambda x: abs(sec - x[1]))
    qlen = dict(grid)[name]
    return name, qlen

def _midi_to_name(m):
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    m = int(round(m))
    return f"{names[m%12]}{(m//12)-1}"

def transcribe_to_json(path, bpm=100, time_sig="4/4"):
    y, sr = _load_audio(path)
    times, f0 = _f0_torchcrepe(y, sr)
    seg = _segment_notes(times, f0)
    notes_json = []
    for t0, t1, hz in seg:
        midi = _hz_to_midi(hz)
        pname = _midi_to_name(midi)
        dur = t1 - t0
        fig, _ = _quantize_len(dur, bpm=bpm)
        notes_json.append({"pitch": pname, "duration": fig})
    return {"tempo": bpm, "time_signature": time_sig, "notes": notes_json}
