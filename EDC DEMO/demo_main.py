from flask import Flask, render_template, request, redirect
import runpy
import json
import os


app = Flask(__name__)

@app.route('/', methods = ['POST','GET'])
@app.route('/index', methods = ['POST','GET'])

def index():
    if request.method == "POST":
        flowstring = "".join(list(request.form.keys()))
        return render_template('request_form.html',value=flowstring)
    else:
        return render_template('index.html')



@app.route('/request_form', methods = ['POST','GET'])
def request_form():
    if request.method == "POST":
        req_first_name = request.form['firstname']
        req_last_name = request.form['lastname']
        req_ssn = request.form['ssn']
        req_flow = request.form['flow']
        responsedoc = run_sdk(req_first_name,req_last_name,req_ssn,req_flow)
        return responsedoc
        # return render_template('response_page.html')
    else:
        return "ISSUES"

def run_sdk(fname,lname,ssn,flow):
    reqdoc = build_request(fname,lname,ssn,flow)
    with open('PythonSDK/Endpoint/current_request.json', 'w') as outfile:
        json.dump(reqdoc, outfile)
    runpy.run_path('PythonSDK/Endpoint/app2.py')

    with open(os.path.join(os.path.dirname(__file__), "PythonSDK/Endpoint/current_response2.json")) as f:
	    respdoc = json.load(f)

    response = respdoc[0]
    return response

def build_request(fname,lname,ssn,flow):
    reqdict = {"FirstName":fname,"LastName":lname,'SSN':ssn,'FlowDesign':flow}
    requestdoc={"Request":{"RequestInfo":reqdict}}
    return requestdoc

if __name__ == "__main__":
    app.run(debug=True)