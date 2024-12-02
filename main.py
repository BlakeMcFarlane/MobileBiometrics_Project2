''' Imports '''
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import get_images
import get_landmarks
import performance_plots
from get_metrics import get_roc, get_det, plot_confusion_matrix, compute_eer, compute_authentication_accuracy

from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
import pandas as pd

''' Import classifier '''
from sklearn.neighbors import KNeighborsClassifier as knn, NearestCentroid
from sklearn.naive_bayes import GaussianNB as gnb


''' Load the data and their labels '''
image_directory = './Caltech Faces Dataset'
X, y = get_images.get_images(image_directory)

''' Get distances between face landmarks in the images '''
features, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 68, True)

''' Matching and Decision '''
clf = ORC(knn())
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)

matching_scores = clf.predict_proba(X_test)

gen_scores = []
imp_scores = []
classes = clf.classes_
matching_scores = pd.DataFrame(matching_scores, columns=classes)

for i in range(len(y_test)):    
    scores = matching_scores.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])
    
# Plot ROC and DET curves
get_roc(gen_scores, imp_scores)
get_det(gen_scores, imp_scores)

# Compute EER and Authentication Accuracy
eer, eer_threshold = compute_eer(gen_scores, imp_scores)
compute_authentication_accuracy(gen_scores, imp_scores, eer_threshold)

# Plot Confusion Matrix
plot_confusion_matrix(clf, X_test, y_test)

performance_plots.performance(gen_scores, imp_scores, 'performance', 100)
