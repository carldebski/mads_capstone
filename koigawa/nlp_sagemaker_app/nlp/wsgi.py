# This code was not written by Capstone project members
# The code copied entirely from below cookie cutter code provided by AWS
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/wsgi.py
# Licensed under the Apache-2.0 license

import predictor as myapp

# This is just a simple wrapper for gunicorn to find your app.
# If you want to change the algorithm file, simply change "predictor" above to the
# new file.

app = myapp.app
