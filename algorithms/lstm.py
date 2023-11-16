import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from algorithms.hasuraRequest import set_algorithm_result

def lstm_anomaly_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    
    # Etiketler ve özelliklerin ayrılması
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = data.drop(['tag', 'time'], axis=1)

    # Veri normalizasyonu
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # LSTM model yapısını oluşturma
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(scaled_features.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Verilerin yeniden şekillendirilmesi
    reshaped_features = np.reshape(scaled_features, (scaled_features.shape[0], scaled_features.shape[1], 1))

    # Modeli eğit
    model.fit(reshaped_features, scaled_features, epochs=2, batch_size=32, shuffle=True)

    # Tahminlerin yapılması ve yeniden yapılandırma hatasının hesaplanması
    predictions = model.predict(reshaped_features)
    mse = np.mean(np.power(scaled_features - predictions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    model_predictions = (mse > threshold).astype(int)

    # Performans metriklerinin hesaplanması
    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(true_labels, model_predictions, zero_division=0)
    recall = recall_score(true_labels, model_predictions, zero_division=0)
    f1 = f1_score(true_labels, model_predictions, zero_division=0)

    # Sonuçların döndürülmesi
    return set_algorithm_result({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, algorithm_settings_id)
