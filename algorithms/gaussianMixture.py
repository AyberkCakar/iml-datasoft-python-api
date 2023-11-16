from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from algorithms.hasuraRequest import set_algorithm_result
from algorithms.normalization import normalization_data

def gmm_outlier_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data.drop(['tag', 'time'], axis=1))

    # GMM modeli
    gmm = GaussianMixture(n_components=3, covariance_type='full')
    gmm.fit(features)
    log_prob = gmm.score_samples(features)

    # Aykırı değer tespiti için eşik değerinin belirlenmesi
    threshold = np.percentile(log_prob, 5)  # Alt %5'lik olasılık
    model_predictions = (log_prob < threshold).astype(int)

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
        #'log_prob': log_prob.tolist(),
        #'is_outlier': model_predictions.tolist()
    }

    return set_algorithm_result(result, algorithm_settings_id)
