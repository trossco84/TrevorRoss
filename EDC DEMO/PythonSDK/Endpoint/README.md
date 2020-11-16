# SMARTS RESTful SDK for Python 3 Examples

## INTRODUCTION ##
The Sparkling Logic SMARTS&trade; demo project's decision service may be accessed through a Sparkling Logic provided cloud.
This document provides an overview of a Python 3 based example to invoke the demo decision service.

## KEY COMPONENTS ##


The example includes the following Sparkling Logic components.

-   slsdk-0.2-py3-none-any.whl

    Python Package for the SDK for interaction with the Sparkling Logic Decision Service. The SDK is a standard Python package and can be installed using pip or pipenv.
    
-   app.py

    Demo for a session based decision service interaction.
    
-   app2.py

    Demo for a session less / token based decision service interaction

-   app3.py

    Demo for a task flow service interaction

## SETUP ##
Run the following commands in the example directory.
-   pipenv install ../Sdk/slsdk-0.2-py3-none-any.whl

    This will install the SDK as part of the example.
	Alternatively you can also install the package using pip instead of pipenv.
	

## USAGE ##

In order to make a connection the following information is needed.
-   A valid username and password for a registered SMARTS user
-   Deployment Access ID and Key

    The Access ID and Key should be kept at secure at all times.

The required credential information is configured with a `smarts.conf.json` setting files.
You must alter the `smarts.conf.json` file and supply the required information.
```bash
pipenv run python app.py
```
The example will print a message for the connection, the status of the processed document after evaluation and then a disconnect message.
, which will send a connection request to the decision service. A small status message will be displayed when the connection has been completed.

__Note:__
The workspace, project name and name of the decision have been coded in the example `smarts.conf.json` file for convenience purposes. If you are a registered user and have access to a demo workspace then your credentials and tenant specific access ID and key can be used. If your demo workspace is not the default “Top/Def” workspace, then you need to change the default settings in the configuration file.
If you are only registered on the Sparkling Logic evaluation cloud as part of the evaluation tenant then you need to request the access ID and key.

## API OVERVIEW ##
The decision service is a REST services with a JSON based payload. This section highlights some of the code sections in the example and goes through the 3 main REST calls:

-   Connect
-   Evaluate
-   Disconnect



### MAKING A CONNECTION ###
The connection call requires the basic connection parameters as well as the security tokens.

```python
from slsdk.rest.client import *
# Required parameters for connect call.
config = SlRestClientConfig(parameters['ServiceUrl'])
client = SlRestClient(config, parameters['AppId'], parameters['AppKey'])

# Connect to the decision service.
request = SlRestDeploymentConnectionRequest(parameters['Username'], parameters['Password'], parameters['Workspace'], parameters['DeploymentId'])
connection = client.connect(request)
if (connection.success is True):
	print("Successfully connected")
else:
	raise "Connection failure: " + connection.errorMessage

```	

The applicationId, key, userId, and pwd should be obtained from a secure source or from user’s input or other secure mechanisms.
A connection request is created with a workspace ("Top/Def") and deployment indentification ("Auto quote project"). These uniquely identify the deployment definition of the demo and will be different for your scenario.

The SDK will create an RFC 2104 HMAC compliant security hash of the information provided as is required by the decision service. 

The response is encapsulated by the SlRestDeploymentConnectionResponse class
When a connection is successful the header will contain the session ID to use for subsequent calls.

```python
sessionId = connection.SessionId
```

### EVALUATING A DECISION ###
After you have made a successful connection and have obtained a session ID, you can continue to make evaluation requests for any decision that is part of the deployment definition.
The example contains a single a set of sample documents in JSON format, located in the `data` folder.  You can invoke the decision service for evaluation as follows:

```python
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
	raise "Evaluation failure: " + evaluation.errorMessage

# Process the results
for doc in evaluation.documents:
	print("Processed: " + doc['ContactInformation']['FirstName'])
```	

The call to "Evaluate" is passed in a session ID (obtained from the connection call) as well as a unique identifier for the decision to invoke. In this example the decision "Auto insurance quote".

The response is encapsulated by the SlRestDeploymentEvaluationResponse class
When a evaluation is successful the response will contain the documents as processed by the decision.

### DISCONNECT FROM THE SERVICE ###
To disconnect from the service you only need to provide the session id and security information as follows:

```python
# Disconnect the session
request = SlRestDeploymentDisconnectionRequest(sessionId)
disconnected = client.disconnect(request)
if (disconnected.success is True):
	print("Successfully disconnected")
else:
	raise "Disconnection failure: " + disconnected.errorMessage
```

## USING ACCESS TOKENS ##

Instead of using credentials and the connect / evaluate / disconnect approach shown above, you can also invoke the decision services using special secure access tokens. Some of the advantage of using access tokens:

-   Fewer REST calls to the decision service

    Once you have obtained a valid access token you can use this token directly for evaluation requests without the need to make a connection or to perform a disconnect.
-   No session and connection management.

    Connections (and their corresponding sessions) are short lived. This requires you to managed connections if  there is a chance a session may timeout in between subsequent evaluation requests. Access tokens are long lived and can be refreshed if they have expired. You can configure the security constraints associated with access tokens in the SMARTS management environment.
-   No session affinity.

    Access tokens are completely sessionless, which means you can have load balancers without session affinity configurations and access tokens will work across a distributed deployment environment.

### API OVERVIEW ###
The decision service is a REST services with a JSON based payload. This section highlights some of the code sections in the example and goes through the 2 main REST calls:

-	RequestAccessToken (optional if you already obtained a token)
-	Evaluate

#### REQUESTING AN ACCESS TOKEN ####
The access token request call requires the basic connection parameters as well as the security tokens.

```python
# Obtain an access token for the decision service.
# Alternatively you can use an already created token from a persistent store.
request = SlRestDeploymentAccessTokenRequest(parameters['Username'], parameters['Password'], parameters['Workspace'], parameters['DeploymentId'])
connection = client.requestAccessToken(request)
if (connection.success is True):
	print("Successfully retrieved access token")
else:
	raise "Token request failure: " + connection.errorMessage

```

Similar as with a standard connection you specify the workspace and deployment identifier as part of the request. Additionally you can specify a specific deployment release and / or metric category as part of the request.

Once you received a successful response you can obtain the access token from the response.

```python
# Retrieve the token from the token response
accessToken = connection.getDeploymentAccessToken()
```

__Note:__ A str() call will provide you with a string representation of the token, which could be used as part of secure persistence environments. You can contstruct token from the string representation. In particular with long lived tokens you can avoid the call to request a token and use a token that is quickly retrieved from a secure location.

#### EVALUATING A DECISION ####
When you have a valid access token you can use this token directly as part of an evaluation request.
The example contains a single sample document in JSON format.  You can invoke the decision service for evaluation as follows:
```python
# Invoke the decision service with the example documents.
# Note: In this example we evaluate all documents in a single call.
# You could also evaluate them one by one and / or in parallel.
request = SlRestDecisionEvaluationRequest(None, accessToken, "Auto insurance quote")
request.documents = documents
evaluation = client.evaluate(request)
# Check the evaluation response.
if (evaluation.success is True):
	print("Successfully evaluated requests")
else:
	raise "Evaluation failure: " + evaluation.errorMessage

# Process the results
for doc in evaluation.documents:
	print("Processed: " + doc['ContactInformation']['FirstName'])
```

The call to `Evaluate` is passed an evaluation request. The request is constructed with an access token as well as a unique identifier for the decision to invoke. In this example the decision "Auto insurance quote".
When a evaluation is successful the body will contain the documents as processed by the decision.

__Note:__ There is no need to disconnect as the access token approach is sessionless. An access token can be persisted and retrieved from secure persistence layers to avoid a request for access tokens altogether (as long as the token is still valid).
