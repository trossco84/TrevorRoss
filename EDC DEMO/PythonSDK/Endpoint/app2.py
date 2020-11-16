import os
import json
import pandas as pd
from slsdk.rest.client import *

parameters = None
# Read the connection / example configuration parameters
with open(os.path.join(os.path.dirname(__file__), "smarts.conf.json")) as f:
	parameters = json.load(f)
# Read all test applications from file
documents = None
with open(os.path.join(os.path.dirname(__file__), "current_request.json")) as f:
	documents = json.load(f)

# Required parameters for connect call.
config = SlRestClientConfig(parameters['ServiceUrl'])
client = SlRestClient(config, parameters['AppId'], parameters['AppKey'])

# Obtain an access token for the decision service.
# Alternatively you can use an already created token from a persistent store.
request = SlRestDeploymentAccessTokenRequest(parameters['Username'], parameters['Password'], parameters['Workspace'], parameters['DeploymentId'])
connection = client.requestAccessToken(request)
if (connection.success is True):
	print("Successfully retrieved access token")
else:
	raise Exception("Token request failure: " + connection.errorMessage)


accessToken = connection.getDeploymentAccessToken()

# Invoke the decision service with the example documents.
# Note: In this example we evaluate all documents in a single call.
# You could also evaluate them one by one and / or in parallel.
request = SlRestDecisionEvaluationRequest(None, accessToken,parameters['DecisionId'])
request.documents = documents
evaluation = client.evaluate(request)
# Check the evaluation response.
if (evaluation.success is True):
	print("Successfully evaluated requests")
else:
	print()
	print("Details:")
	print(evaluation.details)
	print()
	print("Documents:")
	print(evaluation.documents)
	print()
	print(str(evaluation))
	raise Exception("Evaluation failure: " + evaluation.errorMessage)

doc_list = evaluation.documents
with open('PythonSDK/Endpoint/current_response2.json', 'w') as outfile:
    json.dump(doc_list, outfile)

