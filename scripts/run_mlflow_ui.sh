#!/bin/bash
# Script to launch MLflow UI

echo "Starting MLflow UI on port 5000..."
echo "Access at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

mlflow ui --port 5000
