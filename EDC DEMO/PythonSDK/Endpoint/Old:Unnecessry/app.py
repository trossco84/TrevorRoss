import os
import json
from slsdk.rest.client import *

parameters = None
# Read the connection / example configuration parameters
with open(os.path.join(os.path.dirname(__file__), "smarts.conf.json")) as f:
	parameters = json.load(f)

# Read all test applications from file
documents = None
with open(os.path.join(os.path.dirname(__file__), "byte_request.json")) as f:
	documents = json.load(f)

# Required parameters for connect call.
config = SlRestClientConfig(parameters['ServiceUrl'])
client = SlRestClient(config, parameters['AppId'], parameters['AppKey'])

# Connect to the decision service.
request = SlRestDeploymentConnectionRequest(parameters['Username'], parameters['Password'], parameters['Workspace'], parameters['DeploymentId'])
connection = client.connect(request)
if (connection.success is True):
	print("Successfully connected")
else:
	raise Exception("Connection failure: " + connection.errorMessage)


sessionId = connection.sessionId

# Invoke the decision service with the example documents.
# Note: In this example we evaluate all documents in a single call.
# You could also evaluate them one by one and / or in parallel.
request = SlRestDecisionEvaluationRequest(sessionId, None, "Auto insurance quote")
request.documents = documents
evaluation = client.evaluate(request)
# Check the evaluation response.
if (evaluation.success is True):
	print("Successfully evaluated requests")
else:
	raise Exception("Evaluation failure: " + evaluation.errorMessage)

# Process the results
for doc in evaluation.documents:
	print("Processed: " + doc['ContactInformation']['FirstName'])
	

# Disconnect the session
request = SlRestDeploymentDisconnectionRequest(sessionId)
disconnected = client.disconnect(request)
if (disconnected.success is True):
	print("Successfully disconnected")
else:
	raise Exception("Disconnection failure: " + disconnected.errorMessage)
