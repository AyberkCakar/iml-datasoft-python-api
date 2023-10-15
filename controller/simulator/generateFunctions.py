
import numpy as np

def generate_audio_data(time_interval, start, x_factor):
    frequency = 440  # Temel frekans (A4 notası)
    num_samples = int(time_interval)
    
    time = np.linspace(start, start + time_interval, num_samples)
    sine_wave = np.sin(2 * np.pi * frequency * time)
    
    audio_data = sine_wave * x_factor
    
    return time, audio_data

def generate_temperature_data(time_interval, start, x_factor, temperature_range):
    num_samples = int(time_interval)
    
    time = np.linspace(start, start + time_interval, num_samples)
    temperatures = np.random.uniform(*temperature_range, num_samples) * x_factor  # Sıcaklık aralığı içinde rastgele sıcaklık değerleri
    
    return time, temperatures


def generate_vibration_data(time_interval, start, x_factor):
    num_samples = int(time_interval)
    
    time = np.linspace(start, start + time_interval, num_samples)
    vibration_data = x_factor * np.sin(time)

    return time, vibration_data