from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from algorithms.hasuraRequest import set_algorithm_result
from algorithms.normalization import normalization_data

def xgboost_outlier_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data.drop(['tag', 'time'], axis=1))

    # XGBoost modeli
    xgb_model = XGBClassifier()
    xgb_model.fit(features, true_labels)

    # Tahminlerin olasılıklarını elde etme
    predictions_proba = xgb_model.predict_proba(features)[:, 1]
    threshold = 0.95
    model_predictions = (predictions_proba > threshold).astype(int)

    # Performans metrikleri
    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(true_labels, model_predictions)
    recall = recall_score(true_labels, model_predictions)
    f1 = f1_score(true_labels, model_predictions)

    return set_algorithm_result({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, algorithm_settings_id)
