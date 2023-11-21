import pandas as pd
from algorithms.abod import abod_outlier_detection
from algorithms.autoencoder import autoencoder_outlier_detection
from algorithms.baggingOutlier import feature_bagging_outlier_detection
from algorithms.cblof import cblof_outlier_detection
from algorithms.dbscanOutlierDetection import dbscan_outlier_detection
from algorithms.deepAutoencoders import deep_autoencoder_outlier_detection
from algorithms.ensemleMethods import ensemble_outlier_detection
from algorithms.gaussianMixture import gmm_outlier_detection
from algorithms.gru import gru_anomaly_detection
from algorithms.hasuraRequest import fetch_algorithm, fetch_real_dataset, fetch_simulator_dataset, fetch_algorithm_setting
from algorithms.hbos import hbos_outlier_detection
from algorithms.isolationForest import isolation_forest
from algorithms.kmeans import kmeans_outlier_detection_with_metrics
from algorithms.knn import knn_outlier_detection
from algorithms.localOutlierFactor import lof_outlier_detection
from algorithms.lstm import lstm_anomaly_detection
from algorithms.lstmAutoencoder import lstm_autoencoder
from algorithms.oneClassSVM import oneclass_svm_outlier_detection
from algorithms.pca import pca_outlier_detection
from algorithms.rnn import rnn_anomaly_detection
from algorithms.robustCovariance import robust_covariance_outlier_detection_with_metrics
from algorithms.rrcf import robust_random_cut_forest
from algorithms.xgboost import xgboost_outlier_detection

failure_types = []


def call_algorithm(algorithm_name, algorithm_settings_id, algorithmId, fetched_data, sensor_types):
    algorithm_functions = {
        'Isolation Forest': isolation_forest,
        'LSTM Autoencoder': lstm_autoencoder,
        'Local Outlier Factor': lof_outlier_detection,
        'KNN': knn_outlier_detection,
        'Autoencoder Outlier Detection': autoencoder_outlier_detection,
        'ABOD Outlier Detection': abod_outlier_detection,
        'Feature Bagging Outlier Detection': feature_bagging_outlier_detection,
        'Ensemble Outlier Detection': ensemble_outlier_detection,
        'K-Means': kmeans_outlier_detection_with_metrics,
        'Robust Covariance Outlier Detection': robust_covariance_outlier_detection_with_metrics,
        'PCA Outlier Detection': pca_outlier_detection,
        'GMM Outlier Detection': gmm_outlier_detection,
        'HOBS Outlier Detection': hbos_outlier_detection,
        'Deep Autoencoder Outlier Detection': deep_autoencoder_outlier_detection,
        'CBLOF Outlier Detection': cblof_outlier_detection,
        'XGBoost Outlier Detection': xgboost_outlier_detection,
        'LSTM Anomaly Detection': lstm_anomaly_detection,
        'RNN Anomaly Detection': rnn_anomaly_detection,
        'GRU Anomaly Detection': gru_anomaly_detection,
        'Robust Random Cut Forest': robust_random_cut_forest,
        # 'dbscan_outlier_detection': dbscan_outlier_detection,
        # 'oneclass_svm_outlier_detection': oneclass_svm_outlier_detection,
    }

    return algorithm_functions.get(algorithm_name, lambda *args: None)(algorithm_settings_id, algorithmId, fetched_data, sensor_types)


def select_algorithm(algorithm_results):
    algorithm_settings_id = algorithm_results.get('algorithm_setting_id')

    algorithm_id = algorithm_results.get('algorithm_id')
    fetched_algorithm = fetch_algorithm(algorithm_id)

    algorithm = fetched_algorithm.get('data').get('algorithms_by_pk')
    algorithmName = algorithm.get('algorithmName')
    algorithmId = algorithm.get('id')

    fetced_algorithm_setting = fetch_algorithm_setting(
        algorithm_settings_id).get('data').get('algorithm_settings_by_pk')

    sensor_types = fetced_algorithm_setting.get('sensorTypes').split(',')

    try:
        if fetced_algorithm_setting.get('simulatorId') != None:
            fetched_data = fetch_simulator_dataset(fetced_algorithm_setting.get(
                'simulatorId')).get('data').get('datasets')[0].get('result')
        else:
            fetched_data = fetch_real_dataset(fetced_algorithm_setting.get(
                'realDatasetId')).get('data').get('datasets')[0].get('result')

        return call_algorithm(algorithmName, algorithm_settings_id, algorithmId, fetched_data, sensor_types)
    except Exception as exc:
        print(algorithmName, exc)
