import asyncio
from dotenv import load_dotenv
from python_graphql_client import GraphqlClient
import os
load_dotenv()

IML_DATASOFT_HASURA_ENDPOINT = os.getenv("IML_DATASOFT_HASURA_ENDPOINT")
IML_DATASOFT_HASURA_SECRET_KEY = os.getenv("IML_DATASOFT_HASURA_SECRET_KEY")

headers = {
    'x-hasura-admin-secret': IML_DATASOFT_HASURA_SECRET_KEY,
}


def fetch_algorithm(algorithmId):
    client = GraphqlClient(endpoint=IML_DATASOFT_HASURA_ENDPOINT,
                           headers=headers, verify=True)

    query = """
        query getAlgorithm($algorithmId: Int!) {
            algorithms_by_pk(id: $algorithmId) {
                algorithmName
                id
            }
        }
     """
    variables = {"algorithmId": algorithmId}

    try:
        return asyncio.run(client.execute_async(query=query, headers=headers, variables=variables))
    except Exception as err:
        print("An error occurred:", err)
        return None


def fetch_algorithm_setting(algorithm_settings_id):
    client = GraphqlClient(endpoint=IML_DATASOFT_HASURA_ENDPOINT, verify=True)

    query = """
        query getAlgorithmSetting($algorithmSettingsId: Int!) {
            algorithm_settings_by_pk(id: $algorithmSettingsId) {
                realDatasetId
                simulatorId
                sensorTypes
            }
        }
     """
    variables = {"algorithmSettingsId": algorithm_settings_id}

    try:
        return asyncio.run(client.execute_async(query=query, headers=headers, variables=variables))
    except Exception as err:
        print("An error occurred:", err)
        return None


def fetch_real_dataset(realDatasetId):
    client = GraphqlClient(endpoint=IML_DATASOFT_HASURA_ENDPOINT, verify=True)

    query = """
        query getRealDataset($realDatasetId: Int!) {
            datasets(where: {realDatasetId: {_eq: $realDatasetId}}) {
                result
                id
            }
        }
     """

    variables = {"realDatasetId": realDatasetId}

    try:
        return asyncio.run(client.execute_async(query=query, headers=headers, variables=variables))
    except Exception as err:
        print("An error occurred:", err)
        return None


def fetch_simulator_dataset(simulatorId):
    client = GraphqlClient(endpoint=IML_DATASOFT_HASURA_ENDPOINT, verify=True)

    query = """
        query getSimulatorDataset($simulatorId: Int!) {
            datasets(where: {simulatorId: {_eq: $simulatorId}}) {
                result
                id
            }
        }
     """

    variables = {"simulatorId": simulatorId}

    try:
        return asyncio.run(client.execute_async(query=query, headers=headers, variables=variables))
    except Exception as err:
        print("An error occurred:", err)
        return None


def update_algorithm_result(result_data, algorithm_settings_id, algorithm_id):
    client = GraphqlClient(endpoint=IML_DATASOFT_HASURA_ENDPOINT, verify=True)

    query = """
        mutation updateAlgorithmResult($algorithmId: Int!, $algorithmSettingId: Int!, $result: jsonb!) {
            update_algorithm_results(where: {algorithmId: {_eq: $algorithmId}, algorithmSettingId: {_eq: $algorithmSettingId}}, _set: {result: $result}) {
                affected_rows
            }
        }
     """

    variables = {
        "result": result_data,
        "algorithmSettingId": algorithm_settings_id,
        "algorithmId": algorithm_id
    }

    try:
        data = asyncio.run(client.execute_async(
            query=query, headers=headers, variables=variables))

        return data.get('data').get(
            'update_algorithm_results').get('affected_rows')
    except Exception as err:
        return None
