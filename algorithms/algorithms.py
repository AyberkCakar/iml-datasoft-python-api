import pandas as pd
from algorithms.abod import abod_outlier_detection
from algorithms.autoencoder import autoencoder_outlier_detection
from algorithms.baggingOutlier import feature_bagging_outlier_detection
from algorithms.dbscanOutlierDetection import dbscan_outlier_detection
from algorithms.ensemleMethods import ensemble_outlier_detection
from algorithms.gaussianMixture import gmm_outlier_detection
from algorithms.hasuraRequest import fetch_algorithm, fetch_real_dataset, fetch_simulator_dataset
from algorithms.isolationForest import isolation_forest
from algorithms.kmeans import kmeans_outlier_detection_with_metrics
from algorithms.knn import knn_outlier_detection
from algorithms.localOutlierFactor import lof_outlier_detection
from algorithms.lstmAutoencoder import lstm_autoencoder
from algorithms.oneClassSVM import oneclass_svm_outlier_detection
from algorithms.pca import pca_outlier_detection
from algorithms.robustCovariance import robust_covariance_outlier_detection_with_metrics

failure_types = []

def select_algorithm(algorithm_settings):
    algorithm_settings_id = algorithm_settings.get('id')

    fetched_algorithm = fetch_algorithm(algorithm_settings.get('algorithm_id')).get('data').get('algorithms_by_pk').get('algorithmName')

    fetched_data = [];
    
    if algorithm_settings.get('simulator_id') != None:
        fetched_data = fetch_simulator_dataset(algorithm_settings.get('simulator_id')).get('data').get('datasets')[0].get('result')
    else:
        fetched_data =fetch_real_dataset(algorithm_settings.get('real_dataset_id')).get('data').get('datasets')[0].get('result')


    if (fetched_algorithm == 'IsolationForest'):
        return isolation_forest(algorithm_settings_id, fetched_data)
    #elif (fetched_algorithm == 'LSTM_Autoencoder'):
        #return lstm_autoencoder(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'lof_outlier_detection'):
        return lof_outlier_detection(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'dbscan_outlier_detection'):
        return dbscan_outlier_detection(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'knn_outlier_detection'):
        return knn_outlier_detection(algorithm_settings_id, fetched_data)
    #elif (fetched_algorithm == 'oneclass_svm_outlier_detection'):
        #return oneclass_svm_outlier_detection(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'autoencoder_outlier_detection'):
        return autoencoder_outlier_detection(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'abod_outlier_detection'):
        return abod_outlier_detection(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'feature_bagging_outlier_detection'):
        return feature_bagging_outlier_detection(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'ensemble_outlier_detection'):
        return ensemble_outlier_detection(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'kmeans_outlier_detection_with_metrics'):
        return kmeans_outlier_detection_with_metrics(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'robust_covariance_outlier_detection_with_metrics'):
        return robust_covariance_outlier_detection_with_metrics(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'pca_outlier_detection'):
        return pca_outlier_detection(algorithm_settings_id, fetched_data)
    elif (fetched_algorithm == 'gmm_outlier_detection'):
        return gmm_outlier_detection(algorithm_settings_id, fetched_data)
    