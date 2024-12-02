import warnings
warnings.filterwarnings("ignore")

import get_images
import get_landmarks
import performance_plots
from get_metrics import get_roc, get_det, plot_confusion_matrix, compute_eer, compute_authentication_accuracy

from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
import pandas as pd
import os
import numpy as np
import librosa
import random

# Define a function to extract MFCC features from a voice file
def extract_mfcc_features(wav_file, sr=16000, n_mfcc=13):
    y, sr = librosa.load(wav_file, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Compute mean and standard deviation of MFCCs
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    return np.concatenate((mfcc_mean, mfcc_std))

# Load the data and their labels
image_directory = './Caltech Faces Dataset'
X_face, y = get_images.get_images(image_directory)

# Get distances between face landmarks in the images
X_face, y = get_landmarks.get_landmarks(X_face, y, 'landmarks/', 68, False)

# Load voice data per person and align with face images
voice_directory = './Voice Data'  # Adjust the path if necessary
person_dirs = sorted([d for d in os.listdir(voice_directory) if os.path.isdir(os.path.join(voice_directory, d))])

# Build a dictionary mapping person_id to list of voice files
person_voice_files = {}
for person_dir in person_dirs:
    person_id = person_dir  # 'person_1', 'person_2', etc.
    person_path = os.path.join(voice_directory, person_dir)
    wav_files = [f for f in os.listdir(person_path) if f.endswith('.wav')]
    person_voice_files[person_id] = [os.path.join(person_path, f) for f in wav_files]

# Align voice samples with face images
X_voice = []
for idx, y_label in enumerate(y):
    person_id = y_label  # Assuming y contains labels like 'person_1', 'person_2', etc.
    voice_files = person_voice_files[person_id]
    # Shuffle the voice files once per person if you want random selection
    random.seed(42)  # For reproducibility
    random.shuffle(voice_files)
    # Select a voice file (use modulo indexing to handle different counts)
    voice_file = voice_files[idx % len(voice_files)]
    mfcc = extract_mfcc_features(voice_file)
    X_voice.append(mfcc)

X_voice = np.array(X_voice)

# Now, the lengths should match
assert len(X_voice) == len(X_face) == len(y), "Number of samples must match between face and voice datasets"

# Split data into training and test sets while keeping modalities aligned
X_face_train, X_face_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_face, y, range(len(y)), test_size=0.33, random_state=42)

X_voice_train = X_voice[idx_train]
X_voice_test = X_voice[idx_test]

# Classifier for Face Modality
clf_face = ORC(KNN())
clf_face.fit(X_face_train, y_train)

# Classifier for Voice Modality
clf_voice = ORC(KNN())
clf_voice.fit(X_voice_train, y_train)

# Predict probabilities for Face Modality
matching_scores_face = clf_face.predict_proba(X_face_test)

# Predict probabilities for Voice Modality
matching_scores_voice = clf_voice.predict_proba(X_voice_test)

# Fuse scores by averaging
matching_scores_fused = (matching_scores_face + matching_scores_voice) / 2.0

# Extract genuine and impostor scores from fused scores
gen_scores = []
imp_scores = []
classes = clf_face.classes_  # Assuming both classifiers have the same classes
matching_scores_df = pd.DataFrame(matching_scores_fused, columns=classes)

for i in range(len(y_test)):
    scores = matching_scores_df.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])

# Performance evaluation using fused scores
get_roc(gen_scores, imp_scores)
get_det(gen_scores, imp_scores)

# Compute EER and Authentication Accuracy
eer, eer_threshold = compute_eer(gen_scores, imp_scores)
compute_authentication_accuracy(gen_scores, imp_scores, eer_threshold)

# Plot Confusion Matrix
# Ensure labels are numpy arrays
y_test = np.array(y_test)
y_pred_fused = matching_scores_df.idxmax(axis=1).to_numpy()

# Use class labels from your classifier or define them directly
classes = clf_face.classes_  # Or classes = np.unique(y_test)

# Plot Confusion Matrix using the modified function
plot_confusion_matrix(y_test, y_pred_fused, classes)


# Optional: Plot performance using your existing function
performance_plots.performance(gen_scores, imp_scores, 'Score-Level Fusion Performance', 100)
