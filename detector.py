import numpy as np
import librosa
import joblib
import os
import pickle
from pathlib import Path

class ScreamDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.latest_result = {"scream_detected": False, "confidence": 0.0}
        
        # Try to load model and scaler if they exist
        try:
            model_dir = Path(__file__).parent / 'models'
            
            # Create models directory if it doesn't exist
            if not model_dir.exists():
                model_dir.mkdir(parents=True)
            
            # Load scaler if it exists
            scaler_path = model_dir / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load model if it exists
            model_path = model_dir / 'model.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    
        except Exception as e:
            print(f"Could not load model files: {e}")
            # Will use rule-based detection as fallback
    
    def extract_features(self, audio):
        """Extract audio features from raw audio data"""
        # Make sure audio is not empty and has correct shape
        if audio.size == 0:
            return np.zeros(38)  # Return empty feature vector
        
        # Ensure we have enough audio data
        if len(audio) < 2048:
            audio = np.pad(audio, (0, 2048 - len(audio)))
        
        # Extract standard audio features
        features = []
        
        # MFCCs - Mel-frequency cepstral coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
        features.extend([np.mean(mfcc) for mfcc in mfccs])
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=22050)[0]
        features.append(np.mean(spectral_centroid))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=22050)[0]
        features.append(np.mean(spectral_bandwidth))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=22050)[0]
        features.append(np.mean(spectral_rolloff))
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.append(np.mean(zcr))
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features.append(np.mean(rms))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=22050)
        features.extend([np.mean(chroma_bin) for chroma_bin in chroma])
        
        # Compute spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=22050)
        features.append(np.mean(contrast))
        
        # Compute tempo and beat strength
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=22050)
        features.append(tempo)
        features.append(len(beat_frames))
        
        # Add some statistical features
        features.append(np.std(rms))  # Variability in energy
        features.append(np.max(rms) - np.min(rms))  # Range of energy
        
        # Make sure we have exactly 38 features (matching the expected number)
        if len(features) > 38:
            features = features[:38]
        while len(features) < 38:
            features.append(0.0)
        
        return np.array(features)
    
    def detect(self, audio):
        """Detect if the audio contains a scream"""
        # Extract features from audio
        features = self.extract_features(audio)
        
        # If we have a trained model and scaler, use them
        if self.model is not None and self.scaler is not None:
            try:
                # Scale features
                scaled_features = self.scaler.transform([features])[0]
                
                # Predict with model
                prediction = self.model.predict_proba([scaled_features])[0]
                is_scream = bool(prediction[1] > 0.5)  # Assuming class 1 is scream
                confidence = float(prediction[1])
                
                self.latest_result = {
                    "scream_detected": is_scream, 
                    "confidence": confidence
                }
                
                return self.latest_result
            except Exception as e:
                print(f"Error in model prediction: {e}")
                # Fall back to rule-based detection
        
        # Rule-based detection as fallback
        # Calculate energy and other features to detect screams
        energy = np.mean(np.abs(audio))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=22050)[0])
        
        # Normalize the features to a 0-1 range for consistency
        norm_energy = min(energy * 5, 2.0)  # Adjust multiplier based on expected values
        norm_centroid = min(spectral_centroid / 5000, 2.0)  # Normalize to expected range
        
        # High intensity and high frequency content are common in screams
        # This is a simplified rule - a proper ML model would do better
        intensity_factor = norm_energy * 0.7
        frequency_factor = norm_centroid * 0.3
        combined_score = intensity_factor + frequency_factor
        
        # Determine if it's a scream
        is_scream = bool(combined_score > 1.0)
        confidence = float(min(combined_score / 2.0, 1.0))  # Scale to 0-1
        
        self.latest_result = {
            "scream_detected": is_scream, 
            "confidence": confidence
        }
        
        return self.latest_result
    
    def get_latest_result(self):
        """Return the latest detection result"""
        return self.latest_result
    
    def save_model(self, model, scaler):
        """Save a trained model and scaler"""
        try:
            model_dir = Path(__file__).parent / 'models'
            
            # Create models directory if it doesn't exist
            if not model_dir.exists():
                model_dir.mkdir(parents=True)
            
            # Save scaler
            scaler_path = model_dir / 'scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save model
            model_path = model_dir / 'model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            self.model = model
            self.scaler = scaler
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False