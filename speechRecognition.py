import sounddevice as sd
import numpy as np
import wave
import librosa
from keras.models import load_model  # type: ignore
import speech_recognition as sr
from sklearn.preprocessing import StandardScaler
import os

model_path = os.path.join('Interview bot', 'model.keras')
model = load_model(model_path)

def scale_data(X):
    reshaped_X = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(reshaped_X)
    return scaled_X.reshape(X.shape)

def record_audio(filename, duration=5):
    fs = 16000  
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())
    print(f"Audio saved to {filename}")

def extract_mfcc_features(audio, sr, n_mfcc=40, max_len=40):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.vstack((mfcc, mfcc_delta, mfcc_delta2))  
    
    if features.shape[1] > max_len:
        features = features[:, :max_len]
    else:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    
    return features

def predict_emotion(audio_file, model):
    y, sr = librosa.load(audio_file, sr=16000, duration=3)
    features = extract_mfcc_features(y, sr)
    
    features = np.expand_dims(features, axis=0)
    features = scale_data(features)
    
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions)
    
    return predicted_class

def capture_audio_and_analyze():
    audio_filename = 'user_response.wav'
    record_audio(audio_filename, duration=5)

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

    predicted_class = predict_emotion(audio_filename, model)
    emotion_labels = ['angry', 'fearful', 'happy', 'neutral', 'sad']

    return text, emotion_labels[predicted_class]

if __name__ == "__main__":
    capture_audio_and_analyze()
