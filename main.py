import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
import librosa  # Library for audio feature extraction
import os

print("hi")

# Function to extract voice features for multiple files per identity
def extract_voice_features(identity, voice_folder):
    features_list = []
    max_files_per_identity = {
        0: 49, 1: 49, 2: 49, 3: 49, 4: 49, 5: 49, 6: 49,
        7: 40, 8: 34
    }
    
    # Determine the number of voice files for the current identity
    num_files = max_files_per_identity.get(identity, 0)  # Default to 0 if identity is not in the dict
    
    for i in range(num_files):  # Iterate over each file from 0 to num_files
        voice_file = os.path.join(voice_folder, f"{identity}_09_{i}.wav")
        
        if os.path.exists(voice_file):
            try:
                voice_data, sr = librosa.load(voice_file, sr=None)
                mfcc_features = librosa.feature.mfcc(voice_data, sr=sr, n_mfcc=13)
                mean_mfcc = np.mean(mfcc_features.T, axis=0)  # Average MFCC features
                features_list.append(mean_mfcc)  # Add MFCC features to the list
            except Exception as e:
                print(f"Error loading {voice_file}: {str(e)}. Skipping this file.")
        else:
            print(f"Warning: Voice file {voice_file} not found.")
    
    if features_list:
        # Average the features across all the voice files for this identity
        aggregated_features = np.mean(features_list, axis=0)
        print(f"Successfully extracted and averaged voice features for identity {identity}")
        return aggregated_features
    else:
        # If no files are found, use zeros as placeholder
        print(f"Warning: No valid voice files found for identity {identity}. Using zeros as placeholder.")
        return np.zeros(13)

class Evaluator:
    def __init__(self, num_thresholds, genuine_scores, impostor_scores, plot_title, epsilon=1e-12):
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        mean_genuine = np.mean(self.genuine_scores)
        mean_impostor = np.mean(self.impostor_scores)
        std_genuine = np.std(self.genuine_scores)
        std_impostor = np.std(self.impostor_scores)
        x = mean_genuine - mean_impostor
        y = np.sqrt((std_genuine ** 2 + std_impostor ** 2) / 2)
        return x / (y + self.epsilon)

    def get_EER(self, FPR, FNR):
        differences = np.abs(np.array(FPR) - np.array(FNR))
        min_index = np.argmin(differences)
        return (FPR[min_index] + FNR[min_index]) / 2

    def calculate_metrics(self, FPR, TPR):
        # True Negative Rate (TNR) = 1 - FPR
        TNR = 1 - FPR
        
        # False Negative Rate (FNR) = 1 - TPR
        FNR = 1 - TPR

        # Accuracy calculation: (TP + TN) / (Total)
        accuracy = (TPR + TNR) / 2

        # Equal Error Rate (EER)
        EER = self.get_EER(FPR, FNR)

        return accuracy, FPR, FNR, TPR, TNR, EER

    def print_metrics(self, accuracy, FPR, FNR, TPR, TNR, EER):
        print(f"d-prime: {self.get_dprime():.4f}")
        print(f"Accuracy: {accuracy.mean():.4f}")
        print(f"False Positive Rate (FPR): {FPR.mean():.4f}")
        print(f"False Negative Rate (FNR): {FNR.mean():.4f}")
        print(f"True Positive Rate (TPR): {TPR.mean():.4f}")
        print(f"True Negative Rate (TNR): {TNR.mean():.4f}")
        print(f"Equal Error Rate (EER): {EER:.4f}")

    def plot_score_distribution(self):
        plt.figure()
        plt.hist(self.genuine_scores, bins=50, color='green', lw=2, histtype='step', label='Genuine', hatch='//')
        plt.hist(self.impostor_scores, bins=50, color='red', lw=2, histtype='step', label='Impostor', hatch='\\')
        plt.xlim([-0.05, 1.05])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper left', fontsize=10)
        plt.xlabel('Scores', fontsize=12, weight='bold')
        plt.ylabel('Frequency', fontsize=12, weight='bold')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(f'Score Distribution Plot\nd-prime= {self.get_dprime():.2f}\nSystem {self.plot_title}', fontsize=15, weight='bold')
        plt.savefig(f'score_distribution_plot_({self.plot_title}).png', dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_det_curve(self, FPR, FNR):
        EER = self.get_EER(FPR, FNR)
        plt.figure()
        plt.plot(FPR, FNR, lw=2, color='blue', label='DET Curve')
        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100, label='EER')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel('FPR', fontsize=12, weight='bold')
        plt.ylabel('FNR', fontsize=12, weight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.title(f'DET Curve\nEER= {EER:.5f}\nSystem {self.plot_title}', fontsize=15, weight='bold')
        plt.savefig(f'DET_curve_({self.plot_title}).png', dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_roc_curve(self, FPR, TPR):
        plt.figure()
        plt.plot(FPR, TPR, lw=2, color='orange', label='ROC Curve')
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel('FPR', fontsize=12, weight='bold')
        plt.ylabel('TPR', fontsize=12, weight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.title(f'ROC Curve\nSystem {self.plot_title}', fontsize=15, weight='bold')
        plt.savefig(f'ROC_curve_({self.plot_title}).png', dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

# Function to combine face and voice features
def combine_features(face_features, voice_features):
    # Concatenate face and voice features
    combined_features = np.concatenate([face_features, voice_features])
    return combined_features

# Load face data (from Code 1)
X = np.load("X-68-Caltech.npy")
y = np.load("y-68-Caltech.npy")
num_identities = y.shape[0]

# Extract face features (as done in original code)
face_features = []
for k in range(num_identities):
    person_k = X[k]
    features_k = []
    for i in range(person_k.shape[0]):
        for j in range(person_k.shape[0]):
            p1 = person_k[i, :]
            p2 = person_k[j, :]
            features_k.append(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))
    face_features.append(features_k)
face_features = np.array(face_features)

# Load voice features for all identities
voice_folder = "Voice Data"
voice_features = []

for i in range(len(y)):
    identity = y[i]
    features = extract_voice_features(identity, voice_folder)
    voice_features.append(features)

voice_features = np.array(voice_features)

# Combine face and voice features for multimodal system
multimodal_features = np.array([combine_features(face_features[i], voice_features[i]) for i in range(num_identities)])

# Define classifier (SVM in this case)
clf = svm.SVC(kernel='rbf', class_weight='balanced', probability=True)

# Initialize lists to store genuine and impostor scores for both systems
genuine_scores_face_only = []
impostor_scores_face_only = []
genuine_scores_multimodal = []
impostor_scores_multimodal = []

# Evaluate the system
for i in range(len(y)):
    # Face-only system
    query_face_X = face_features[i, :]
    query_y = y[i]
    template_face_X = np.delete(face_features, i, 0)
    template_y = np.delete(y, i)
    
    clf.fit(template_face_X, template_y)
    
    # Predict probabilities for face-only system
    y_prob_face = clf.predict_proba(query_face_X.reshape(1, -1))[0]
    genuine_scores_face_only.append(y_prob_face[1])
    impostor_scores_face_only.append(y_prob_face[0])

    # Multimodal system (Face + Voice)
    query_multimodal_X = multimodal_features[i, :]
    template_multimodal_X = np.delete(multimodal_features, i, 0)

    clf.fit(template_multimodal_X, template_y)
    
    # Predict probabilities for multimodal system
    y_prob_multimodal = clf.predict_proba(query_multimodal_X.reshape(1, -1))[0]
    genuine_scores_multimodal.append(y_prob_multimodal[1])
    impostor_scores_multimodal.append(y_prob_multimodal[0])

# Convert scores to numpy arrays
genuine_scores_face_only = np.array(genuine_scores_face_only)
impostor_scores_face_only = np.array(impostor_scores_face_only)
genuine_scores_multimodal = np.array(genuine_scores_multimodal)
impostor_scores_multimodal = np.array(impostor_scores_multimodal)

# Initialize evaluators for both systems
evaluator_face_only = Evaluator(num_thresholds=200, genuine_scores=genuine_scores_face_only, impostor_scores=impostor_scores_face_only, plot_title='Face-Only System')
evaluator_multimodal = Evaluator(num_thresholds=200, genuine_scores=genuine_scores_multimodal, impostor_scores=impostor_scores_multimodal, plot_title='Multimodal System (Face + Voice)')

# Evaluate Face-Only System
FPR_face, TPR_face, _ = metrics.roc_curve(np.concatenate([np.ones_like(genuine_scores_face_only), np.zeros_like(impostor_scores_face_only)]),
                                          np.concatenate([genuine_scores_face_only, impostor_scores_face_only]))
FNR_face = 1 - TPR_face
accuracy_face, FPR_face, FNR_face, TPR_face, TNR_face, EER_face = evaluator_face_only.calculate_metrics(FPR_face, TPR_face)

# Evaluate Multimodal System
FPR_multimodal, TPR_multimodal, _ = metrics.roc_curve(np.concatenate([np.ones_like(genuine_scores_multimodal), np.zeros_like(impostor_scores_multimodal)]),
                                                      np.concatenate([genuine_scores_multimodal, impostor_scores_multimodal]))
FNR_multimodal = 1 - TPR_multimodal
accuracy_multimodal, FPR_multimodal, FNR_multimodal, TPR_multimodal, TNR_multimodal, EER_multimodal = evaluator_multimodal.calculate_metrics(FPR_multimodal, TPR_multimodal)

# Print metrics for both systems
print("\nFace-Only System Metrics:")
evaluator_face_only.print_metrics(accuracy_face, FPR_face, FNR_face, TPR_face, TNR_face, EER_face)

print("\nMultimodal System (Face + Voice) Metrics:")
evaluator_multimodal.print_metrics(accuracy_multimodal, FPR_multimodal, FNR_multimodal, TPR_multimodal, TNR_multimodal, EER_multimodal)

# Plot score distributions and curves for both systems
evaluator_face_only.plot_score_distribution()
evaluator_face_only.plot_det_curve(FPR_face, FNR_face)
evaluator_face_only.plot_roc_curve(FPR_face, TPR_face)

evaluator_multimodal.plot_score_distribution()
evaluator_multimodal.plot_det_curve(FPR_multimodal, FNR_multimodal)
evaluator_multimodal.plot_roc_curve(FPR_multimodal, TPR_multimodal)

print("bye")
