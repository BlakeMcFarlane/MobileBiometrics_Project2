import os
import cv2  # For image processing
import librosa  # For audio processing
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern  # Use scikit-image for LBP

# Directories for local datasets
image_data_dir = r"C:\Users\happy\OneDrive - University of South Florida\Classes\Fall 24\CAP 4103\Project\MobileBiometrics_Project2\08"  
# Directory for face image files
voice_data_dir = r"C:\Users\happy\OneDrive - University of South Florida\Classes\Fall 24\CAP 4103\Project\MobileBiometrics_Project2\Voices"  # Directory for voice .wav files

# ---------------------------
# Load Face Images and Apply LBP
# ---------------------------
image_files = [f for f in os.listdir(image_data_dir) if f.endswith('.png') or f.endswith('.jpg')][:5]

# Process each image for LBP
for file in image_files:
    img_path = os.path.join(image_data_dir, file)
    
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))  # Resize image (100x100)
    
    # Display image for verification
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(f"Original Image: {file}")
    plt.show()
    
    # Compute LBP feature extraction using skimage's local_binary_pattern
    radius = 1  # radius of neighborhood
    n_points = 8 * radius  # number of points to consider
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')  # Apply LBP
    
    # Display the LBP image
    plt.figure()
    plt.imshow(lbp, cmap='gray')
    plt.title(f"LBP of {file}")
    plt.show()

# ---------------------------
# Load Voice Files and Convert to Spectrograms, Apply LBP
# ---------------------------
voice_files = [f for f in os.listdir(voice_data_dir) if f.endswith('.wav')][:5]

# Process each voice file into spectrogram and apply LBP
for file in voice_files:
    audio_path = os.path.join(voice_data_dir, file)
    
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Generate mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)  # Convert to decibel scale
    
    # Resize spectrogram to match image size
    spectrogram_resized = cv2.resize(spectrogram_db, (100, 100))
    
    # Display spectrogram for verification
    plt.figure()
    librosa.display.specshow(spectrogram_resized, sr=sr, x_axis='time', y_axis='mel')
    plt.title(f"Spectrogram of {file}")
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    
    # Apply LBP on the spectrogram (use the same LBP method as above)
    lbp_spectrogram = local_binary_pattern(spectrogram_resized, n_points, radius, method='uniform')
    
    # Display the LBP spectrogram
    plt.figure()
    plt.imshow(lbp_spectrogram, cmap='gray')
    plt.title(f"LBP of Spectrogram for {file}")
    plt.show()
