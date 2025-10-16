import librosa
import numpy as np
import json
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = "audios"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return jsonify({"status": "Servidor activo 🚀"})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400

    f = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)

    try:
        # --- Cargar audio ---
        y, sr = librosa.load(path, sr=16000, mono=True)

        # --- Detección de pitch (pYIN) ---
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            frame_length=2048,
            sr=sr
        )

        # --- Convertir a notas ---
        times = librosa.times_like(f0, sr=sr)
        notes = []

        last_note = None
        start_time = None

        for t, pitch, voiced in zip(times, f0, voiced_flag):
            if voiced and pitch is not None:
                current_note = librosa.hz_to_note(pitch)
                if last_note is None:
                    last_note = current_note
                    start_time = t
                elif current_note != last_note:
                    notes.append({
                        "note": last_note,
                        "start": float(start_time),
                        "end": float(t),
                        "duration": float(t - start_time)
                    })
                    last_note = current_note
                    start_time = t
            elif last_note is not None:
                notes.append({
                    "note": last_note,
                    "start": float(start_time),
                    "end": float(t),
                    "duration": float(t - start_time)
                })
                last_note = None
                start_time = None

        if last_note is not None and start_time is not None:
            notes.append({
                "note": last_note,
                "start": float(start_time),
                "end": float(times[-1]),
                "duration": float(times[-1] - start_time)
            })

        # --- Calcular tempo aproximado ---
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # --- Filtrar duraciones y limpiar ---
        notes = [n for n in notes if n["duration"] > 0.05]

        return jsonify({
            "tempo": round(float(tempo), 2),
            "notes": notes,
            "count": len(notes)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
