import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalization_data(data):
    features = data[['amplitude', 'vibration', 'temperature']]
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return pd.DataFrame(normalized_features, columns=['amplitude', 'vibration', 'temperature'])
