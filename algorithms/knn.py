from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from algorithms.hasuraRequest import set_algorithm_result
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from algorithms.normalization import normalization_data

def knn_outlier_detection(algorithm_settings_id, fetched_data, n_neighbors=5):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    # KNN modeli
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(features)
    distances, indices = knn.kneighbors(features)

    # Aykırı değer puanı hesaplama
    outlier_score = np.mean(distances, axis=1)
    threshold = np.percentile(outlier_score, 95)  # Örneğin %95 eşik değeri
    model_predictions = (outlier_score > threshold).astype(int)


    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(true_labels, model_predictions)
    recall = recall_score(true_labels, model_predictions)
    f1 = f1_score(true_labels, model_predictions)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
       # 'outlier_score': outlier_score.tolist(),
        #'is_outlier': model_predictions.tolist()
    }

    return set_algorithm_result(result, algorithm_settings_id)
