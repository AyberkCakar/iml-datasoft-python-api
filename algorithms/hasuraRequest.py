import asyncio
from python_graphql_client import GraphqlClient
import os

DLBAD_HASURA_ENDPOINT = os.getenv("DLBAD_HASURA_ENDPOINT")

def fetch_algorithm(algorithmId):
    client = GraphqlClient(endpoint=DLBAD_HASURA_ENDPOINT, verify=True)

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
        return asyncio.run(client.execute_async(query=query, variables=variables))
    except Exception as err:
        print("An error occurred:", err)
        return None


def fetch_real_dataset(realDatasetId):
    client = GraphqlClient(endpoint=DLBAD_HASURA_ENDPOINT, verify=True)

    query = """
        query getRealDataset($realDatasetId: Int!) {
            datasets(where: {realDatasetId: {_eq: $realDatasetId}}) {
                result
                id
                isRealData
            }
        }
     """

    variables = {"realDatasetId": realDatasetId}

    try:
        return asyncio.run(client.execute_async(query=query, variables=variables))
    except Exception as err:
        print("An error occurred:", err)
        return None


def fetch_simulator_dataset(simulatorId):
    client = GraphqlClient(endpoint=DLBAD_HASURA_ENDPOINT, verify=True)

    query = """
        query getRealDataset($simulatorId: Int!) {
            datasets(where: {simulatorId: {_eq: $simulatorId}}) {
                result
                id
                isRealData
            }
        }
     """

    variables = {"simulatorId": simulatorId}

    try:
        return asyncio.run(client.execute_async(query=query, variables=variables))
    except Exception as err:
        print("An error occurred:", err)
        return None

def set_algorithm_result(result_data, algorithm_settings_id):
    print('data', result_data)
    client = GraphqlClient(endpoint=DLBAD_HASURA_ENDPOINT, verify=True)

    query = """
        mutation INSERT_ALGORITHM_RESULT($object: algorithm_results_insert_input!) {
            insert_algorithm_results_one(object: $object) {
                id
            }
        }
     """

    variables = {
        "object": {
            "result": result_data,
            "algorithmSettingId": algorithm_settings_id
        }
    }

    try:
        data = asyncio.run(client.execute_async(query=query, variables=variables))
        print('result', data)
        id  = data.get('data').get('insert_algorithm_results_one').get('id')
        return id
    except Exception as err:
        print("An error occurred:", err)
        return None
