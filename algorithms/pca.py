from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def pca_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data.drop(['tag', 'time'], axis=1))

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)

    reconstruction_error = np.sum(
        np.square(features - pca.inverse_transform(pca_features)), axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    model_predictions = (reconstruction_error > threshold).astype(int)

    result = {
        # 'reconstruction_error': reconstruction_error.tolist(),
        # 'is_outlier': model_predictions.tolist()
    }

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)
