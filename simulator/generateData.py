from simulator.hasuraRequest import fetch_failure_types, set_simulator_data
from simulator.generateFunctions import generate_sound_data, generate_temperature_data, generate_vibration_data
import numpy as np
import random

failure_types = []


def generate_data(interval_count, simulatorData):
    simulatorId = simulatorData.get('id')
    expectedSoundValue = simulatorData.get('expected_sound_value')
    expectedTemperatureValue = simulatorData.get('expected_temperature_value')
    expectedVibrationValue = simulatorData.get('expected_vibration_value')

    fetched_data = fetch_failure_types(simulatorId)
    if fetched_data is not None:
        if "data" in fetched_data:
            transformed_data = fetched_data["data"]["simulator_parameters"]
            for item in transformed_data:
                failure_type = item.get("failure_type", {})
                transformed_item = {
                    "failureName": failure_type.get("failureName", ""),
                    "id": failure_type.get("id", 0),
                    "period": failure_type.get("period", 0),
                    "soundAnomalyMultiplier": failure_type.get("soundAnomalyMultiplier", 0),
                    "temperatureAnomalyMultiplier": failure_type.get("temperatureAnomalyMultiplier", 0),
                    "timeInterval": failure_type.get("timeInterval", 0),
                    "vibrationAnomalyMultiplier": failure_type.get("vibrationAnomalyMultiplier", 0),
                }
                failure_types.append(transformed_item)

    output_data = {'time': [], 'sound': [],
                   'temperature': [], 'vibration': [], 'tag': []}

    random_numbers = np.random.randint(0, interval_count, 10)

    start = 0

    for i in range(interval_count):
        selected_object = random.choice(failure_types)
        time_interval = selected_object['timeInterval']

        if i in random_numbers:
            sound_anomaly_multiplier = selected_object['soundAnomalyMultiplier']
            temperature_anomaly_multiplier = selected_object['temperatureAnomalyMultiplier']
            vibration_anomaly_multiplier = selected_object['vibrationAnomalyMultiplier']
            tag = selected_object['failureName']
        else:
            sound_anomaly_multiplier = 1
            temperature_anomaly_multiplier = 1
            vibration_anomaly_multiplier = 1
            tag = 'Normal'

        minExpectedTemperatureValue = int(expectedTemperatureValue) - 5
        maxExpectedTemperatureValue = int(expectedTemperatureValue) + 5

        time_points, audio_data = generate_sound_data(
            time_interval, start, sound_anomaly_multiplier, expectedSoundValue)
        time_points, temperature_data = generate_temperature_data(
            time_interval, start, temperature_anomaly_multiplier, (minExpectedTemperatureValue, maxExpectedTemperatureValue))
        time_points, vibration_data = generate_vibration_data(
            time_interval, start, vibration_anomaly_multiplier, expectedVibrationValue)

        output_data['time'].extend(time_points)
        output_data['sound'].extend(audio_data)
        output_data['temperature'].extend(temperature_data)
        output_data['vibration'].extend(vibration_data)
        output_data['tag'].extend([tag] * len(time_points))

        start += time_interval

    return set_simulator_data(output_data, simulatorId)
