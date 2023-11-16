from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from algorithms.hasuraRequest import set_algorithm_result
from algorithms.normalization import normalization_data

def feature_bagging_outlier_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    # LOF algoritması
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof.fit_predict(features)
    lof_scores = -lof.negative_outlier_factor_

    # KNN algoritması
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(features)
    distances, _ = knn.kneighbors(features)
    knn_scores = np.mean(distances, axis=1)

    # Ortalama aykırı değer puanı hesaplama
    average_score = (lof_scores + knn_scores) / 2

    # Eşik değeri belirleme ve aykırı değer tespiti
    threshold = np.percentile(average_score, 95)
    model_predictions = (average_score > threshold).astype(int)
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
       # 'average_score': average_score.tolist(),
        #'is_outlier': model_predictions.tolist()
    }

    return set_algorithm_result(result, algorithm_settings_id)
