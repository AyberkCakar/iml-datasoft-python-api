from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from algorithms.hasuraRequest import set_algorithm_result
from algorithms.normalization import normalization_data

def pca_outlier_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data.drop(['tag', 'time'], axis=1))

    # PCA modeli
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)

    reconstruction_error = np.sum(np.square(features - pca.inverse_transform(pca_features)), axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    model_predictions = (reconstruction_error > threshold).astype(int)

    # Performans metrikleri
    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(true_labels, model_predictions)
    recall = recall_score(true_labels, model_predictions)
    f1 = f1_score(true_labels, model_predictions)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        #'reconstruction_error': reconstruction_error.tolist(),
        #'is_outlier': model_predictions.tolist()
    }

    return set_algorithm_result(result, algorithm_settings_id)
