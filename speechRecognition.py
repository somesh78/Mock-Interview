import sounddevice as sd
import numpy as np
import wave
import librosa
from keras.models import load_model # type: ignore
import speech_recognition as sr
from sklearn.preprocessing import StandardScaler
import os

# Load the trained model
model_path = os.path.join('Interview bot', 'model.keras')
model = load_model(model_path)

# Function to apply StandardScaler for 3D data
def scale_data(X):
    # Reshape the data to 2D: [samples * time_steps, features]
    reshaped_X = X.reshape(-1, X.shape[-1])
    
    # Apply StandardScaler
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(reshaped_X)
    
    # Reshape back to 3D: [samples, time_steps, features]
    return scaled_X.reshape(X.shape)

# Record the audio
def record_audio(filename, duration=5):
    fs = 16000  # Match the training sample rate
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    print("Recording complete.")
    
    # Save as WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())
    print(f"Audio saved to {filename}")

def extract_mfcc_features(audio, sr, n_mfcc=40, max_len=40):
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Stack MFCC, delta, and delta-delta features
    features = np.vstack((mfcc, mfcc_delta, mfcc_delta2))  # Shape: (120, time_steps)
    
    # Ensure consistent length through padding or truncation
    if features.shape[1] > max_len:
        features = features[:, :max_len]
    else:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    
    return features

def predict_emotion(audio_file, model):
    # Load and preprocess audio
    y, sr = librosa.load(audio_file, sr=16000, duration=3)  # Match training sample rate
    
    # Extract features
    features = extract_mfcc_features(y, sr)
    
    # Reshape and standardize
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = scale_data(features)
    
    # Predict emotion
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions)
    
    return predicted_class

def capture_audio_and_analyze():
    audio_filename = 'user_response.wav'
    record_audio(audio_filename, duration=5)

    # Transcribe audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None, None
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the audio.")
            return None, None

    # Predict emotion
    predicted_class = predict_emotion(audio_filename, model)
    emotion_labels = ['angry', 'fearful', 'happy', 'neutral', 'sad']

    return text, emotion_labels[predicted_class]

# Run the capture and analysis
if __name__ == "__main__":
    capture_audio_and_analyze()