import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import scipy.stats as stats


# Load the Olivetti face dataset (facial data)
faces = fetch_olivetti_faces()
X_faces = faces.data  # Flattened facial images (400 samples, 64x64 images)
y_faces = faces.target  # Corresponding labels (40 classes of people)

# Load voice data from .wav files
voice_dir = r"Voice Data"
wav_files = sorted([f for f in os.listdir(voice_dir) if f.endswith('.wav')])

# Define a function to extract MFCC features from a voice file
def extract_mfcc_features(wav_file, sr=16000, n_mfcc=13):
    y, sr = librosa.load(wav_file, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)  # Average across time to get a single feature vector

# Extract MFCC features for each voice file
X_voice = []
for wav_file in wav_files:
    mfcc = extract_mfcc_features(os.path.join(voice_dir, wav_file))
    X_voice.append(mfcc)
X_voice = np.array(X_voice)

# Assuming the same number of voice files as face images and labels
assert len(X_voice) == len(X_faces) == len(y_faces), "Number of samples must match between face and voice datasets"

# Combine face and voice features for multimodal input
X_combined = np.hstack((X_faces, X_voice))  # Concatenate face and voice features

# Split data into training and test sets (80% train, 20% test)
X_train_face, X_test_face, y_train, y_test = train_test_split(X_faces, y_faces, test_size=0.2, random_state=42)
X_train_comb, X_test_comb, _, _ = train_test_split(X_combined, y_faces, test_size=0.2, random_state=42)

# Train classifiers using face-only and multimodal (face+voice) data
clf_face = svm.SVC(kernel='rbf', class_weight='balanced', probability=True)
clf_comb = svm.SVC(kernel='rbf', class_weight='balanced', probability=True)

clf_face.fit(X_train_face, y_train)
clf_comb.fit(X_train_comb, y_train)

# Evaluate performance
y_pred_face = clf_face.predict(X_test_face)
y_pred_comb = clf_comb.predict(X_test_comb)

# Confusion matrix calculation
def calculate_confusion_matrix_metrics(y_true, y_pred, label_name):
    # Calculate the confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    # Calculate metrics for each class and average them (macro average)
    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    
    print(f"\nMetrics for {label_name}:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['macro avg']['precision']:.4f}")

# Example usage:
calculate_confusion_matrix_metrics(y_test, y_pred_face, "Face Only")
calculate_confusion_matrix_metrics(y_test, y_pred_comb, "Face + Voice")

# Get prediction probabilities for ROC curves
y_prob_face = clf_face.predict_proba(X_test_face)
y_prob_comb = clf_comb.predict_proba(X_test_comb)

# Calculate EER and ROC curves
def calculate_eer(y_true, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = 0.5  # Set the threshold to 0.5
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]  # Equal error rate (FPR at EER)
    return eer, eer_threshold


# Assuming binary classification (classifying one person's identity as genuine)
# Modify as needed depending on how your target labels are structured
y_test_binary = (y_test == 1).astype(int)  # Modify for binary EER
eer_face, eer_threshold_face = calculate_eer(y_test_binary, y_prob_face[:, 1])
eer_comb, eer_threshold_comb = calculate_eer(y_test_binary, y_prob_comb[:, 1])

# Print EER values
print(f"EER for Face Only: {eer_face:.4f} (Threshold: {eer_threshold_face:.4f})")
print(f"EER for Face + Voice: {eer_comb:.4f} (Threshold: {eer_threshold_comb:.4f})")

# Plot ROC curves
fpr_face, tpr_face, _ = metrics.roc_curve(y_test_binary, y_prob_face[:, 1])
fpr_comb, tpr_comb, _ = metrics.roc_curve(y_test_binary, y_prob_comb[:, 1])

plt.figure()
plt.plot(fpr_face, tpr_face, label=f'Face Only (EER={eer_face:.2f})')
plt.plot(fpr_comb, tpr_comb, label=f'Face + Voice (EER={eer_comb:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Genuine vs Impostor score distribution
def plot_score_distribution(genuine_scores, impostor_scores, title):
    plt.hist(genuine_scores, bins=50, color='green', lw=2, histtype='step', label='Genuine')
    plt.hist(impostor_scores, bins=50, color='red', lw=2, histtype='step', label='Impostor')
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

# Function to calculate d-prime
def calculate_d_prime(genuine_scores, impostor_scores):
    # Calculate hit rate and false alarm rate
    hit_rate = np.mean(genuine_scores > 0.5)  # Threshold set to 0.5
    false_alarm_rate = np.mean(impostor_scores > 0.5)  # Threshold set to 0.5

    # Apply epsilon to avoid extreme values that would cause NaN in Z calculation
    epsilon = 1e-5  # Small value to bound rates between 0 and 1
    hit_rate = np.clip(hit_rate, epsilon, 1 - epsilon)
    false_alarm_rate = np.clip(false_alarm_rate, epsilon, 1 - epsilon)

    # Apply z-score transformation
    z_hit_rate = stats.norm.ppf(hit_rate)
    z_false_alarm_rate = stats.norm.ppf(false_alarm_rate)

    # Calculate d-prime
    d_prime = z_hit_rate - z_false_alarm_rate
    return d_prime



# Genuine vs Impostor scores for Face Only
genuine_scores_face = y_prob_face[y_test_binary == 1, 1]
impostor_scores_face = y_prob_face[y_test_binary == 0, 1]

# Genuine vs Impostor scores for Face + Voice
genuine_scores_comb = y_prob_comb[y_test_binary == 1, 1]
impostor_scores_comb = y_prob_comb[y_test_binary == 0, 1]

# Calculate d-prime for both models
d_prime_face = calculate_d_prime(genuine_scores_face, impostor_scores_face)
d_prime_comb = calculate_d_prime(genuine_scores_comb, impostor_scores_comb)

# Print d-prime values
print(f"D-prime for Face Only: {d_prime_face:.4f}")
print(f"D-prime for Face + Voice: {d_prime_comb:.4f}")

# Print the number of genuine and imposter scores for Face Only and Face + Voice
num_genuine_face = len(genuine_scores_face)
num_imposter_face = len(impostor_scores_face)
num_genuine_comb = len(genuine_scores_comb)
num_imposter_comb = len(impostor_scores_comb)

print(f"Face Only: Genuine = {num_genuine_face}, Imposter = {num_imposter_face}")
print(f"Face + Voice: Genuine = {num_genuine_comb}, Imposter = {num_imposter_comb}")

# Plot the score distributions
plot_score_distribution(genuine_scores_face, impostor_scores_face, 'Face Only Genuine vs Impostor')
plot_score_distribution(genuine_scores_comb, impostor_scores_comb, 'Face + Voice Genuine vs Impostor')
