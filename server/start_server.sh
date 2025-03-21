#!/bin/bash

# Install dependencies if needed
/opt/anaconda3/bin/pip install -r requirements.txt

# Start the Flask server
/opt/anaconda3/bin/python server.py
