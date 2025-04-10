from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for rendering
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import time
import threading
import queue
import pyaudio
import wave

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global variables for audio recording
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
CHUNK = 1024
recording = False
audio_queue = queue.Queue()
scream_detected = False
confidence_level = 0.0
audio_data = []
voice_intensity = []

# Load the model once when the app starts
model = load_model('models/audio_classification.keras')

def extract_features(audio_data, sr=RATE):
    """Extract MFCC features from audio data"""
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features = mfccs_mean.reshape(1, 40, 1, 1).astype('float32')
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def extract_features_from_file(file_path):
    """Extract features from an audio file"""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features = mfccs_mean.reshape(1, 40, 1, 1).astype('float32')
        return features, audio, sr
    except Exception as e:
        print(f"Error extracting features from file: {e}")
        return None, None, None

def audio_recording_thread():
    """Thread function for audio recording"""
    global recording, audio_data, voice_intensity
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    while recording:
        try:
            data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
            audio_queue.put(data)
            audio_data.extend(data)
            
            # Keep only last 2 seconds for analysis
            if len(audio_data) > RATE * 2:
                audio_data = audio_data[-RATE * 2:]
            
            # Update voice intensity for visualization
            chunk_intensity = np.abs(data) 
            voice_intensity = chunk_intensity[::20]  # Downsample for visualization
            
        except Exception as e:
            print(f"Error recording audio: {e}")
            break
    
    stream.stop_stream()
    stream.close()
    p.terminate()

def prediction_thread():
    """Thread function for continuous prediction"""
    global recording, scream_detected, confidence_level, audio_data
    
    while recording:
        if len(audio_data) >= RATE:  # Have at least 1 second of audio
            try:
                # Process the current audio buffer
                features = extract_features(np.array(audio_data))
                if features is not None:
                    prediction = model.predict(features, verbose=0)
                    confidence_level = float(prediction[0][1])  # Assuming class 1 is scream
                    scream_detected = confidence_level > 0.5
            except Exception as e:
                print(f"Error in prediction: {e}")
        
        # Wait before next prediction
        time.sleep(0.3)

@app.route('/')
def index():
    """Render the main template"""
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start audio recording for live detection"""
    global recording, audio_data, scream_detected, confidence_level, voice_intensity
    
    if not recording:
        recording = True
        audio_data = []
        scream_detected = False
        confidence_level = 0.0
        voice_intensity = []
        
        # Start recording and prediction threads
        threading.Thread(target=audio_recording_thread).start()
        threading.Thread(target=prediction_thread).start()
        
        return jsonify({"status": "success", "message": "Recording started"})
    else:
        return jsonify({"status": "error", "message": "Already recording"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop audio recording"""
    global recording
    
    if recording:
        recording = False
        return jsonify({"status": "success", "message": "Recording stopped"})
    else:
        return jsonify({"status": "error", "message": "Not recording"})

@app.route('/get_status')
def get_status():
    """Get current detection status for live mode"""
    global recording, scream_detected, confidence_level, voice_intensity
    
    return jsonify({
        "is_recording": recording,
        "scream_detected": scream_detected,
        "confidence": confidence_level,
        "voice_intensity": voice_intensity.tolist() if isinstance(voice_intensity, np.ndarray) else []
    })

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    """Analyze uploaded audio file for screams"""
    if 'audio-file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})
    
    file = request.files['audio-file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features and make prediction
        features, audio, sr = extract_features_from_file(filepath)
        
        if features is not None:
            prediction = model.predict(features, verbose=0)
            confidence = float(prediction[0][1])  # Assuming class 1 is scream
            scream_detected = confidence > 0.5
            
            # Generate waveform for visualization
            waveform_filename = f"{os.path.splitext(filename)[0]}_waveform.png"
            waveform_path = os.path.join('static', waveform_filename)
            plt.figure(figsize=(10, 2))
            plt.title("Audio Waveform")
            plt.plot(audio)
            plt.axis('off')  # Hide axes
            plt.tight_layout()
            plt.savefig(waveform_path)
            plt.close()
            
            # Prepare voice intensity data for visualization
            # Downsample for reasonable visualization data size
            intensity_data = np.abs(audio)
            step = max(1, len(intensity_data) // 1000)
            downsampled_intensity = intensity_data[::step]
            
            return jsonify({
                "status": "success",
                "scream_detected": scream_detected,
                "confidence": confidence,
                "waveform_image": f"/static/{waveform_filename}",
                "voice_intensity": downsampled_intensity.tolist()
            })
        else:
            return jsonify({"status": "error", "message": "Error processing audio file"})
            
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return jsonify({"status": "error", "message": f"Analysis error: {str(e)}"})

@app.route('/templates/<path:path>')
def send_template(path):
    """Serve template files"""
    return render_template(path)

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)