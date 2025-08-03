#!/bin/bash

# Upgrade pip and install build tools first
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt
