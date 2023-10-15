from hasuraRequest import fetch_failure_types
from generateFunctions import generate_audio_data, generate_temperature_data, generate_vibration_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

failure_types = []

def generate_data(interval_count, failure_type_ids):

    fetched_data = fetch_failure_types(failure_type_ids)
    if fetched_data is not None:
        if "data" in fetched_data:
            failure_types = fetched_data["data"]["failure_types"]

    output_data = {'Zaman': [], 'Amplitude': [],'Temperature': [],'Vibration': [], 'Etiket': []}
    
    random_numbers = np.random.randint(0, interval_count, 10)  # 0 ile 1000 arasında rastgele 10 sayı
   
    start = 0  # Başlangıç değeri
    
    for i in range(interval_count):
        selected_object = random.choice(failure_types)
        time_interval = selected_object['time_interval']  # Rastgele time interval
        
        if i in random_numbers:
            sound_anomaly_multiplier= selected_object['sound_anomaly_multiplier']
            temperature_anomaly_multiplier= selected_object['temperature_anomaly_multiplier']
            vibration_anomaly_multiplier= selected_object['vibration_anomaly_multiplier']
            etiket = selected_object['failure_name']
        else:
            sound_anomaly_multiplier= 1
            temperature_anomaly_multiplier= 1
            vibration_anomaly_multiplier= 1
            etiket = 'Normal'
            
        time_points, audio_data = generate_audio_data(time_interval, start, sound_anomaly_multiplier)
        time_points, temperature_data = generate_temperature_data(time_interval, start, temperature_anomaly_multiplier, (30, 40))
        time_points, vibration_data = generate_vibration_data(time_interval, start, vibration_anomaly_multiplier)

        output_data['Zaman'].extend(time_points)
        output_data['Amplitude'].extend(audio_data)
        output_data['Temperature'].extend(temperature_data)
        output_data['Vibration'].extend(vibration_data)
        output_data['Etiket'].extend([etiket] * len(time_points))
        
        start += time_interval  # Önceki end değeri şimdi yeni start değeri oluyor
    
    df = pd.DataFrame(output_data)
    return output_data;
    
    """

    excel_filename = 'ses_verileri.xlsx'
    df.to_excel(excel_filename, index=False, engine='openpyxl')
    
    print(f"Veriler Excel dosyasına kaydedildi: {excel_filename}")

    # Ses verilerini görselleştirme
    plt.figure(figsize=(10, 6))
    plt.plot(output_data['Zaman'], output_data['Amplitude'])
    plt.title("Üretilen Ses Verileri")
    plt.xlabel("Zaman")
    plt.ylabel("Amplitude")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Sıcaklık verilerini görselleştirme
    plt.figure(figsize=(10, 6))
    plt.plot(output_data['Zaman'], output_data['Temperature'])
    plt.title("Üretilen Sıcaklık Verileri")
    plt.xlabel("Zaman")
    plt.ylabel("Temperature")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Titreşim verilerini görselleştirme
    plt.figure(figsize=(10, 6))
    plt.plot(output_data['Zaman'], output_data['Vibration'])
    plt.title("Üretilen Titreşim Verileri")
    plt.xlabel("Zaman")
    plt.ylabel("Vibration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    """

