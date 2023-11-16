from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

from algorithms.hasuraRequest import set_algorithm_result

def gru_anomaly_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = data.drop(['tag', 'time'], axis=1)

    # Veri normalizasyonu
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features = scaled_features.reshape(scaled_features.shape[0], scaled_features.shape[1], 1)

    # GRU model yapısını oluşturma
    model = Sequential([
        GRU(units=50, return_sequences=True, input_shape=(scaled_features.shape[1], 1)),
        GRU(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Modeli eğit
    model.fit(scaled_features, scaled_features, epochs=2, batch_size=32, shuffle=True)

    # Tahminlerin yapılması ve yeniden yapılandırma hatasının hesaplanması
    predictions = model.predict(scaled_features)
    if predictions.shape[1] == 1:
        predictions = np.expand_dims(predictions, axis=-1)
    
    mse = np.mean(np.power(scaled_features - predictions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    model_predictions = (mse > threshold).astype(int)

    # Performans metriklerinin hesaplanması
    # Performans metriklerinin hesaplanması
    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(true_labels, model_predictions, zero_division=0)
    recall = recall_score(true_labels, model_predictions, zero_division=0)
    f1 = f1_score(true_labels, model_predictions, zero_division=0)

    return set_algorithm_result({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, algorithm_settings_id)
  