import numpy as np


def generate_sound_data(time_interval, start, x_factor, expectedSoundValue):
    frequency = int(expectedSoundValue)
    num_samples = int(time_interval)

    time = np.linspace(start, start + time_interval, num_samples)
    sine_wave = np.sin(2 * np.pi * frequency * time)

    audio_data = sine_wave * x_factor

    return time, audio_data


def generate_temperature_data(time_interval, start, x_factor, expectedTemperatureRange):
    num_samples = int(time_interval)

    time = np.linspace(start, start + time_interval, num_samples)
    temperatures = np.random.uniform(
        *expectedTemperatureRange, num_samples) * x_factor

    return time, temperatures


def generate_vibration_data(time_interval, start, x_factor, expectedVibrationValue):
    num_samples = int(time_interval)
    expectedVibrationRange = (
        expectedVibrationValue - 5, expectedVibrationValue + 5)

    time = np.linspace(start, start + time_interval, num_samples)
    vibration_data = np.random.uniform(
        *expectedVibrationRange, num_samples) * x_factor

    return time, vibration_data
