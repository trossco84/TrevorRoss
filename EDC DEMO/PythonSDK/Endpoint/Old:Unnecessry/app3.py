import os
import json
from time import sleep
from slsdk.rest.client import *

parameters = None
# Read the connection / example configuration parameters
with open(os.path.join(os.path.dirname(__file__), "smarts.conf.json")) as f:
	parameters = json.load(f)

# Required parameters for connect call.
config = SlRestClientConfig(parameters['TaskFlowServiceUrl'])
client = SlRestTaskFlowClient(config, parameters['AppId'], parameters['AppKey'])

# Connect to the decision service.
request = SlRestDeploymentConnectionRequest(parameters['Username'], parameters['Password'], parameters['Workspace'], None)
connection = client.connect(request)
if (connection.success is True):
	print("Successfully connected")
else:
	raise Exception("Connection failure: " + connection.errorMessage)


sessionId = connection.sessionId

# Invoke the decision service with the example documents.
# Note: In this example we evaluate all documents in a single call.
# You could also evaluate them one by one and / or in parallel.
request = SlRestTaskFlowExecutionRequest(sessionId, "Straight Empty")
flowexecution = client.execute(request)

taskflowId = flowexecution.taskflowId

# Check the evaluation response.
if (flowexecution.success is True):
	print("Trigger task flow succeeded")
else:
	raise Exception("Trigger task flow failed: " + flowexecution.errorMessage)

# Wait for the taskflow completion.
# This is just an example, typically you would exist or do some other processing and then 
# check when the taskflow completed.
completed = False
response = None
for i in range(10):
	request = SlRestTaskFlowRequest(sessionId, taskflowId)
	response = client.getStatus(request)
	if (response.status == "Completed"):
		completed = True
		break
	sleep(0.5)

if (completed is True):
	print("Trigger task flow completed succeeded")
else:
	raise Exception("Trigger task flow failed: " + response.status)

# Get the results.
if (completed is True):
	request = SlRestTaskFlowRequest(sessionId, taskflowId)
	results = client.getResults(request)
	if (results.success is True):
		print("Successfully retrieved results")
	else:
		raise Exception("Get results failure: " + results.errorMessage)

# Disconnect the session
request = SlRestDeploymentDisconnectionRequest(sessionId)
disconnected = client.disconnect(request)
if (disconnected.success is True):
	print("Successfully disconnected")
else:
	raise Exception("Disconnection failure: " + disconnected.errorMessage)
