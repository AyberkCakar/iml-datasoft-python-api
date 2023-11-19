import asyncio
from python_graphql_client import GraphqlClient
import os

DLBAD_HASURA_ENDPOINT = os.getenv("DLBAD_HASURA_ENDPOINT")


def fetch_failure_types(simulatorId):
    client = GraphqlClient(endpoint=DLBAD_HASURA_ENDPOINT, verify=True)

    query = """
        query getSimulatorFailureTypes($simulatorId: Int!) {
            simulator_parameters(where: {simulatorId: {_eq: $simulatorId}}) {
                failure_type {
                failureName
                id
                period
                soundAnomalyMultiplier
                temperatureAnomalyMultiplier
                timeInterval
                vibrationAnomalyMultiplier
                }
            }
        }
     """
    variables = {"simulatorId": simulatorId}

    try:
        data = asyncio.run(client.execute_async(
            query=query, variables=variables))
        return data
    except Exception as err:
        print("An error occurred:", err)
        return None


def set_simulator_data(data, simulatorId):
    client = GraphqlClient(endpoint=DLBAD_HASURA_ENDPOINT, verify=True)

    query = """
       mutation INSERT_DATASET($object: datasets_insert_input!) {
            insert_datasets_one(object: $object) {
                id
            }
        }
     """

    variables = {
        "object": {
            "result": data,
            "simulatorId": simulatorId
        }}

    try:
        data = asyncio.run(client.execute_async(
            query=query, variables=variables))
        id = data.get('data').get('insert_datasets_one').get('id')
        return id
    except Exception as err:
        print("An error occurred:", err)
        return None
