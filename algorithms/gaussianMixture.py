from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def gmm_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data, sensor_types)

    gmm = GaussianMixture(n_components=3, covariance_type='full')
    gmm.fit(features)
    log_prob = gmm.score_samples(features)

    threshold = np.percentile(log_prob, 5)
    model_predictions = (log_prob < threshold).astype(int)

    result = {
        # 'log_prob': log_prob.tolist(),
        # 'is_outlier': model_predictions.tolist()
    }

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)
