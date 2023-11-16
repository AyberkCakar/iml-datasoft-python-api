from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from algorithms.hasuraRequest import set_algorithm_result
from algorithms.normalization import normalization_data

def deep_autoencoder_outlier_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data.drop(['tag', 'time'], axis=1))

    # Autoencoder modeli
    input_layer = Input(shape=(features.shape[1],))
    encoded = Dense(100, activation='relu')(input_layer)
    encoded = Dense(50, activation='relu')(encoded)
    decoded = Dense(50, activation='relu')(encoded)
    decoded = Dense(features.shape[1], activation='sigmoid')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(features, features, epochs=20, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)

    # Yeniden yapılandırma hatası
    predictions = autoencoder.predict(features)
    mse = np.mean(np.power(features - predictions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    model_predictions = (mse > threshold).astype(int)

    # Performans metrikleri
    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(true_labels, model_predictions)
    recall = recall_score(true_labels, model_predictions)
    f1 = f1_score(true_labels, model_predictions)

    return set_algorithm_result({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, algorithm_settings_id)
