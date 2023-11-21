from xgboost import XGBClassifier
import pandas as pd
from algorithms.normalization import normalization_data
from algorithms.updateAlgorithmResult import calculate_metrics_and_update_algorithm_result


def xgboost_outlier_detection(algorithm_settings_id, algorithm_id, fetched_data, sensor_types):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data, sensor_types)

    xgb_model = XGBClassifier()
    xgb_model.fit(features, true_labels)

    predictions_proba = xgb_model.predict_proba(features)[:, 1]
    threshold = 0.95
    model_predictions = (predictions_proba > threshold).astype(int)

    return calculate_metrics_and_update_algorithm_result(true_labels, model_predictions, algorithm_settings_id, algorithm_id)
