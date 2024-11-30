import os
import numpy as np
import cv2  # For image processing
import librosa  # For audio processing
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Directories for local datasets
image_data_dir = r"C:\Users\happy\OneDrive - University of South Florida\Classes\Fall 24\CAP 4103\Project\MobileBiometrics_Project2\08"  
# Directory for face image files
voice_data_dir = r"C:\Users\happy\OneDrive - University of South Florida\Classes\Fall 24\CAP 4103\Project\MobileBiometrics_Project2\Voices"  # Directory for voice .wav files

# ---------------------------
# Load Face Images
# ---------------------------
image_files = [f for f in os.listdir(image_data_dir) if f.endswith('.png') or f.endswith('.jpg')][:5]
image_list = []

# Load and preprocess images
for file in image_files:
    img_path = os.path.join(image_data_dir, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # Resize to a consistent size (64x64)
    img_flattened = img.flatten()  # Flatten the image into a 1D array
    image_list.append(img_flattened)

# Convert image list to numpy array
faces = np.array(image_list)

# ---------------------------
# Load Voice Files and Convert to Spectrograms
# ---------------------------
voice_files = [f for f in os.listdir(voice_data_dir) if f.endswith('.wav')][:5]
voice_list = []

# Load and preprocess .wav files (convert to spectrograms)
for file in voice_files:
    audio_path = os.path.join(voice_data_dir, file)
    y, sr = librosa.load(audio_path)  # Load the audio file
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)  # Convert to decibel scale
    spectrogram_resized = cv2.resize(spectrogram_db, (64, 64))  # Resize the spectrogram to match image size
    spectrogram_flattened = spectrogram_resized.flatten()  # Flatten into a 1D array
    voice_list.append(spectrogram_flattened)

# Convert voice list to numpy array
voices = np.array(voice_list)

# ---------------------------
# Combine Faces and Voices
# ---------------------------
# To combine faces and voices, ensure both have the same number of features (flattened array length)
data = np.vstack((faces, voices))  # Stack the face and voice data

# ---------------------------
# Apply PCA
# ---------------------------
n_samples, n_features = data.shape
print(f"Dataset consists of {n_samples} samples, each with {n_features} features.")

n_components = min(n_samples, 10)  # Number of components for PCA
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(data)
data_pca = pca.transform(data)

# PCA transformation completed for both images and voice data
print(f"PCA transformation completed, reduced to {n_components} components.")
