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


def call_algorithm(algorithm_name, algorithm_settings_id, algorithmId, fetched_data):
    algorithm_functions = {
        'IsolationForest': isolation_forest,
        'LSTM_Autoencoder': lstm_autoencoder,
        'lof_outlier_detection': lof_outlier_detection,
        'knn_outlier_detection': knn_outlier_detection,
        'autoencoder_outlier_detection': autoencoder_outlier_detection,
        'abod_outlier_detection': abod_outlier_detection,
        'feature_bagging_outlier_detection': feature_bagging_outlier_detection,
        'ensemble_outlier_detection': ensemble_outlier_detection,
        'kmeans_outlier_detection_with_metrics': kmeans_outlier_detection_with_metrics,
        'robust_covariance_outlier_detection_with_metrics': robust_covariance_outlier_detection_with_metrics,
        'pca_outlier_detection': pca_outlier_detection,
        'gmm_outlier_detection': gmm_outlier_detection,
        'hbos_outlier_detection': hbos_outlier_detection,
        'deep_autoencoder_outlier_detection': deep_autoencoder_outlier_detection,
        'cblof_outlier_detection': cblof_outlier_detection,
        'xgboost_outlier_detection': xgboost_outlier_detection,
        'lstm_anomaly_detection': lstm_anomaly_detection,
        'rnn_anomaly_detection': rnn_anomaly_detection,
        'gru_anomaly_detection': gru_anomaly_detection,
        'robust_random_cut_forest': robust_random_cut_forest,
        # 'dbscan_outlier_detection': dbscan_outlier_detection,
        # 'oneclass_svm_outlier_detection': oneclass_svm_outlier_detection,
    }

    return algorithm_functions.get(algorithm_name, lambda *args: None)(algorithm_settings_id, algorithmId, fetched_data)


def select_algorithm(algorithm_results):
    algorithm_results_id = algorithm_results.get('id')
    algorithm_settings_id = algorithm_results.get('algorithm_setting_id')

    fetched_algorithm = fetch_algorithm(algorithm_results.get(
        'algorithm_id')).get('data').get('algorithms_by_pk')
    algorithmName = fetched_algorithm.get('algorithmName')
    algorithmId = fetched_algorithm.get('id')

    fetced_algorithm_setting = fetch_algorithm_setting(
        algorithm_settings_id).get('data').get('algorithm_settings_by_pk')

    if fetced_algorithm_setting.get('simulatorId') != None:
        fetched_data = fetch_simulator_dataset(fetced_algorithm_setting.get(
            'simulatorId')).get('data').get('datasets')[0].get('result')
    else:
        fetched_data = fetch_real_dataset(fetced_algorithm_setting.get(
            'realDatasetId')).get('data').get('datasets')[0].get('result')

    return call_algorithm(algorithmName, algorithm_settings_id, algorithmId, fetched_data)
