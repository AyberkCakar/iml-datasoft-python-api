import asyncio
from python_graphql_client import GraphqlClient

def fetch_failure_types(failure_type_ids):
    url = "http://localhost:34343/v1/graphql"
    client = GraphqlClient(endpoint=url, verify=True)

    query = """
        query failureTypes($failure_type_ids: [Int!]) {
            failure_types(where: {id: {_in: $failure_type_ids}}) {
                failure_name
                period
                id
                sound_anomaly_multiplier
                temperature_anomaly_multiplier
                time_interval
                vibration_anomaly_multiplier
            }
        }
     """
    variables = {"failure_type_ids": failure_type_ids}

    try:
        data = asyncio.run(client.execute_async(query=query, variables=variables))
        return data
    except Exception as err:
        print("An error occurred:", err)
        return None
