import numpy as np
from pyod.models.combination import aom, moa, average, maximization
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
# Diğer gereken modeller ve metrikler
import pandas as pd
from algorithms.hasuraRequest import set_algorithm_result
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from algorithms.normalization import normalization_data

def ensemble_outlier_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    models = [KNN(), LOF(), ABOD()]
    model_predictions = [model.fit_predict(features) for model in models]

    averaged_predictions = np.mean(model_predictions, axis=0)
    binary_predictions = np.round(averaged_predictions) 

    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions)
    recall = recall_score(true_labels, binary_predictions)
    f1 = f1_score(true_labels, binary_predictions)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    return set_algorithm_result(result, algorithm_settings_id)
