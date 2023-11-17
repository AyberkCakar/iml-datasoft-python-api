from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def knn_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(features)
    distances, indices = knn.kneighbors(features)

    outlier_score = np.mean(distances, axis=1)
    threshold = np.percentile(outlier_score, 95)
    model_predictions = (outlier_score > threshold).astype(int)

    result = {
        # 'outlier_score': outlier_score.tolist(),
        # 'is_outlier': model_predictions.tolist()
    }

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)
