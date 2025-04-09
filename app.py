from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import sounddevice as sd
import librosa
import threading
import time
import os
import tempfile
from werkzeug.utils import secure_filename
from detector import ScreamDetector

app = Flask(__name__)
detector = ScreamDetector()

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
is_recording = False
recording_thread = None
sample_rate = 22050  # Standard sample rate for audio analysis
voice_intensity = []  # Store voice intensity for visualization

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

def record_audio():
    global is_recording, voice_intensity
    buffer_duration = 1.0  # 1 second buffer
    
    while is_recording:
        # Record audio for buffer_duration seconds
        audio_data = sd.rec(int(buffer_duration * sample_rate), 
                            samplerate=sample_rate, 
                            channels=1, 
                            dtype='float32')
        sd.wait()  # Wait until recording is done
        
        if is_recording:  # Check again to avoid processing after stop button
            # Calculate voice intensity for visualization
            # We'll use a downsampled version of the audio data
            downsampled = audio_data[::512].flatten()
            voice_intensity = downsampled.tolist()
            
            # Process the audio
            result = detector.detect(audio_data.flatten())
            print(f"Detection result: {result}")
            # No need to send result to client as we'll use polling

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, recording_thread, voice_intensity
    
    if not is_recording:
        is_recording = True
        voice_intensity = []
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
        return jsonify({"status": "success", "message": "Recording started"})
    
    return jsonify({"status": "error", "message": "Already recording"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    
    if is_recording:
        is_recording = False
        return jsonify({"status": "success", "message": "Recording stopped"})
    
    return jsonify({"status": "error", "message": "Not recording"})

@app.route('/get_status', methods=['GET'])
def get_status():
    # This endpoint allows the frontend to poll for the detection status
    global voice_intensity
    latest_status = detector.get_latest_result()
    
    # Ensure all values are native Python types
    response = {
        "is_recording": bool(is_recording),
        "scream_detected": bool(latest_status.get("scream_detected", False)),
        "confidence": float(latest_status.get("confidence", 0.0)),
        "voice_intensity": voice_intensity
    }
    
    return jsonify(response)

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    # Check if the post request has the file part
    if 'audio-file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['audio-file']
    
    # If user does not select file, browser submits an empty file
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load the audio file with librosa
            audio, sr = librosa.load(filepath, sr=sample_rate, mono=True)
            
            # Calculate voice intensity for visualization
            # We'll downsample the audio to create visualization data
            step = max(1, len(audio) // 1000)  # Ensure we don't have too many points
            downsampled = audio[::step]
            voice_intensity_data = downsampled.tolist()
            
            # Detect scream in the audio file
            # We'll process it in chunks to simulate real-time detection
            chunk_size = sr  # 1 second chunks
            max_confidence = 0.0
            scream_detected = False
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) < chunk_size:  # If last chunk is too small
                    break
                    
                result = detector.detect(chunk)
                if result["confidence"] > max_confidence:
                    max_confidence = result["confidence"]
                
                if result["scream_detected"]:
                    scream_detected = True
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return jsonify({
                "status": "success",
                "scream_detected": bool(scream_detected),
                "confidence": float(max_confidence),
                "voice_intensity": voice_intensity_data
            })
            
        except Exception as e:
            # Clean up in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"status": "error", "message": str(e)}), 500
    
    return jsonify({"status": "error", "message": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)