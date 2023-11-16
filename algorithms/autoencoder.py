import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from algorithms.hasuraRequest import set_algorithm_result
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from algorithms.normalization import normalization_data

def autoencoder_outlier_detection(algorithm_settings_id, fetched_data):
    data = pd.DataFrame(fetched_data)
    true_labels = data['tag'].apply(lambda x: 0 if x == 'Normal' else 1)
    features = normalization_data(data)

    # Özelliklerin ölçeklendirilmesi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    # Autoencoder modeli
    input_layer = Input(shape=(X_scaled.shape[1],))
    encoded = Dense(16, activation='relu')(input_layer)
    decoded = Dense(X_scaled.shape[1], activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
    autoencoder.fit(X_scaled, X_scaled, epochs=20, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)

    # Aykırı değer tespiti
    predictions = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
    threshold = np.percentile(mse, 95)  # Örneğin %95 eşik değeri
    model_predictions = (mse > threshold).astype(int)

    accuracy = accuracy_score(true_labels, model_predictions)
    precision = precision_score(true_labels, model_predictions)
    recall = recall_score(true_labels, model_predictions)
    f1 = f1_score(true_labels, model_predictions)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        #'mse': mse.tolist(),
       # 'is_outlier': model_predictions.tolist()
    }

    return set_algorithm_result(result, algorithm_settings_id)
