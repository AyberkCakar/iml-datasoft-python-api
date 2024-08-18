from simulator.hasuraRequest import fetch_failure_types, set_simulator_data
from simulator.generateFunctions import generate_sound_data, generate_temperature_data, generate_vibration_data
import numpy as np
import random

failure_types = []


def generate_data(simulatorData):
    simulatorId = simulatorData.get('id')
    min_expected_sound_value = simulatorData.get('min_expected_sound_value')
    max_expected_sound_value = simulatorData.get('max_expected_sound_value')
    min_expected_temperature_value = simulatorData.get(
        'min_expected_temperature_value')
    max_expected_temperature_value = simulatorData.get(
        'max_expected_temperature_value')
    min_expected_vibration_value = simulatorData.get(
        'min_expected_vibration_value')
    max_expected_vibration_value = simulatorData.get(
        'max_expected_vibration_value')
    interval_count = simulatorData.get(
        'data_count')
    anomaly_count = simulatorData.get(
        'anomaly_count')

    fetched_data = fetch_failure_types(simulatorId)
    if fetched_data is not None:
        if "data" in fetched_data:
            transformed_data = fetched_data["data"]["simulator_parameters"]
            for item in transformed_data:
                failure_type = item.get("failure_type", {})
                transformed_item = {
                    "failureName": failure_type.get("failureName", ""),
                    "id": failure_type.get("id", 0),
                    "soundAnomalyMultiplier": failure_type.get("soundAnomalyMultiplier", 0),
                    "temperatureAnomalyMultiplier": failure_type.get("temperatureAnomalyMultiplier", 0),
                    "timeInterval": failure_type.get("timeInterval", 0),
                    "vibrationAnomalyMultiplier": failure_type.get("vibrationAnomalyMultiplier", 0),
                }
                failure_types.append(transformed_item)

    anomaly_data = {'time': [], 'sound': [], 'temperature': [], 'vibration': [], 'tag': []}

    random_numbers = np.random.randint(0, interval_count, int(anomaly_count))

    total_anomaly_data = 0

    for _ in random_numbers:
        selected_object = random.choice(failure_types)
        time_interval = selected_object['timeInterval']

        sound_anomaly_multiplier = selected_object['soundAnomalyMultiplier']
        temperature_anomaly_multiplier = selected_object['temperatureAnomalyMultiplier']
        vibration_anomaly_multiplier = selected_object['vibrationAnomalyMultiplier']
        tag = selected_object['failureName']

        time_points, audio_data = generate_sound_data(
            time_interval, 0, sound_anomaly_multiplier, min_expected_sound_value, max_expected_sound_value)
        time_points, temperature_data = generate_temperature_data(
            time_interval, 0, temperature_anomaly_multiplier, min_expected_temperature_value,
            max_expected_temperature_value)
        time_points, vibration_data = generate_vibration_data(
            time_interval, 0, vibration_anomaly_multiplier, min_expected_vibration_value,
            max_expected_vibration_value)

        anomaly_data['time'].append(time_points)
        anomaly_data['sound'].append(audio_data)
        anomaly_data['temperature'].append(temperature_data)
        anomaly_data['vibration'].append(vibration_data)
        anomaly_data['tag'].append([tag] * len(time_points))

        total_anomaly_data += len(time_points)

    normal_data_count = interval_count - total_anomaly_data
    output_data = {'time': [], 'sound': [], 'temperature': [], 'vibration': [], 'tag': []}

    start = 0
    for _ in range(normal_data_count):
        time_interval = 1
        x_factor = 1

        time_points, audio_data = generate_sound_data(
            time_interval, start, x_factor, min_expected_sound_value, max_expected_sound_value)
        time_points, temperature_data = generate_temperature_data(
            time_interval, start, x_factor, min_expected_temperature_value, max_expected_temperature_value)
        time_points, vibration_data = generate_vibration_data(
            time_interval, start, x_factor, min_expected_vibration_value, max_expected_vibration_value)

        output_data['time'].extend(time_points)
        output_data['sound'].extend(audio_data)
        output_data['temperature'].extend(temperature_data)
        output_data['vibration'].extend(vibration_data)
        output_data['tag'].extend(['Normal'] * len(time_points))

        start += time_interval

    for i in range(len(anomaly_data['time'])):
        insert_index = random.randint(0, len(output_data['time']))
        output_data['time'][insert_index:insert_index] = anomaly_data['time'][i]
        output_data['sound'][insert_index:insert_index] = anomaly_data['sound'][i]
        output_data['temperature'][insert_index:insert_index] = anomaly_data['temperature'][i]
        output_data['vibration'][insert_index:insert_index] = anomaly_data['vibration'][i]
        output_data['tag'][insert_index:insert_index] = anomaly_data['tag'][i]

    output_data['time'] = [int(i) for i in range(1, len(output_data['time']) + 1)]

    return set_simulator_data(output_data, simulatorId)
