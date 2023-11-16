from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from algorithms.hasuraRequest import set_algorithm_result
from algorithms.normalization import normalization_data

def kmeans_outlier_detection_with_metrics(algorithm_settings_id, fetched_data, n_clusters=5):
    data = pd.DataFrame(fetched_data)

    # Gerçek etiketler
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Her noktanın kendi küme merkezine olan uzaklığı
    distances = [np.linalg.norm(x-cluster_centers[labels[i]]) for i, x in enumerate(features.values)]

    # Eşik değeri ve aykırı değer tespiti
    threshold = np.percentile(distances, 95)
    model_predictions = (distances > threshold).astype(int)

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
        #'distances': distances,
        #'is_outlier': model_predictions.tolist()
    }

    return set_algorithm_result(result, algorithm_settings_id)