from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from algorithms.hasuraRequest import set_algorithm_result
from algorithms.normalization import normalization_data

def robust_covariance_outlier_detection_with_metrics(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)

    # Gerçek etiketler
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    # Robust Covariance modeli
    robust_cov = EllipticEnvelope(contamination=0.1)
    robust_cov.fit(features)
    predictions = robust_cov.predict(features)

    model_predictions = [1 if x == -1 else 0 for x in predictions]

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
        #'is_outlier': model_predictions
    }

    return set_algorithm_result(result, algorithm_settings_id)