from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def kmeans_outlier_detection_with_metrics(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data, sensor_types)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(features)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    distances = [np.linalg.norm(x-cluster_centers[labels[i]])
                 for i, x in enumerate(features.values)]

    threshold = np.percentile(distances, 95)
    model_predictions = (distances > threshold).astype(int)

    result = {
        # 'distances': distances,
        # 'is_outlier': model_predictions.tolist()
    }

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)
