"# sentiment_Analysis - MLOps Pipeline" 

## Overview

A production-ready machine learning sentiment analysis API that classifies text into 3 sentiment : positive, negative or neutral.  
Build with FastAPI, scikit-learn and docker for easy deployment.  

**Model Type:** TF-IDF+LOgistic Regression  
**Dataset:** IMDB Movie Reviews from kaggle  
**Deployment:** FastAPI + Docker +K8  
***Monitoring:** Prometheus + Grafana  

## Quick Start


### Local Development
```bash

# Install dependencies
make install

# Train model
make train

# Run APPI locally
make serve

# Test endpoint
curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"data": "sample input"}'

