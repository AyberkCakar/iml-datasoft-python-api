import numpy as np


def generate_sound_data(time_interval, start, x_factor, min_expected_sound_value, max_expected_sound_value):
    num_samples = int(time_interval)

    time = np.linspace(start, start + time_interval, num_samples)
    audio_data = np.random.uniform(
        *(min_expected_sound_value, max_expected_sound_value), num_samples) * x_factor

    return time, audio_data


def generate_temperature_data(time_interval, start, x_factor, min_expected_temperature_range, max_expected_temperature_range):
    num_samples = int(time_interval)

    time = np.linspace(start, start + time_interval, num_samples)
    temperatures = np.random.uniform(
        *(min_expected_temperature_range, max_expected_temperature_range), num_samples) * x_factor

    return time, temperatures


def generate_vibration_data(time_interval, start, x_factor, min_expected_vibration_value, max_expected_vibration_value):
    num_samples = int(time_interval)

    time = np.linspace(start, start + time_interval, num_samples)
    vibration_data = np.random.uniform(
        *(min_expected_vibration_value, max_expected_vibration_value), num_samples) * x_factor

    return time, vibration_data
