import warnings
warnings.filterwarnings("ignore")

import get_images
import get_landmarks
import performance_plots
from get_metrics import get_roc, get_det, plot_confusion_matrix, compute_eer, compute_authentication_accuracy

from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm
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
X_face, y = get_landmarks.get_landmarks(X_face, y, 'landmarks/', 5, False)

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

# Combine face and voice features for multimodal input
X = np.hstack((X_face, X_voice))

# Now, the lengths should match
assert len(X_voice) == len(X_face) == len(y), "Number of samples must match between face and voice datasets"

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Matching and Decision - Classifier 1 (k-NN)
clf_knn = ORC(knn())
clf_knn.fit(X_train, y_train)

# KNN scores
matching_scores_knn = clf_knn.predict_proba(X_test)

# Extract genuine and impostor scores
gen_scores = []
imp_scores = []
classes = clf_knn.classes_
matching_scores_df = pd.DataFrame(matching_scores_knn, columns=classes)

for i in range(len(y_test)):
    scores = matching_scores_df.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])

# Performance evaluation
# Plot ROC and DET curves
get_roc(gen_scores, imp_scores)
get_det(gen_scores, imp_scores)

# Compute EER and Authentication Accuracy
eer, eer_threshold = compute_eer(gen_scores, imp_scores)
compute_authentication_accuracy(gen_scores, imp_scores, eer_threshold)

# Plot Confusion Matrix
plot_confusion_matrix(clf_knn, X_test, y_test)

performance_plots.performance(gen_scores, imp_scores, 'performance', 100)
