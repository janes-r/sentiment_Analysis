from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, start_http_server
import time
import logging
from pathlib import Path
import uvicorn
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    SentimentRequest, 
    SentimentResponse, 
    BatchSentimentRequest, 
    BatchSentimentResponse,
    HealthResponse,
    ModelInfoResponse
)
from api.predictor import SentimentPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with metadata
app = FastAPI(
    title="🎭 Sentiment Analysis API",
    description="""
    **Production-ready sentiment analysis API** for movie reviews and general text.
    
    Built with:
    - **Machine Learning**: Logistic Regression with TF-IDF features
    - **Training Data**: IMDB movie reviews dataset (50k samples)
    - **Accuracy**: ~87-90% on test data
    - **Features**: Text preprocessing, batch processing, monitoring
    
    ## Usage Examples:
    
    **Single prediction:**
    ```json
    {
        "text": "This movie is absolutely amazing!"
    }
    ```
    
    **Batch prediction:**
    ```json
    {
        "texts": [
            "Great movie, loved it!",
            "Terrible film, waste of time.",
            "It was okay, nothing special."
        ]
    }
    ```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter(
    'sentiment_predictions_total', 
    'Total number of sentiment predictions made',
    ['sentiment', 'endpoint']
)
prediction_latency = Histogram(
    'sentiment_prediction_duration_seconds', 
    'Time spent on sentiment prediction',
    ['endpoint']
)
error_counter = Counter(
    'sentiment_errors_total', 
    'Total number of prediction errors',
    ['error_type']
)
request_counter = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint']
)

# Global variables
app_start_time = time.time()
predictor = None

# Initialize predictor at module level
def initialize_predictor():
    global predictor
    logger.info("🚀 Initializing Sentiment Analysis API...")
    
    try:
        predictor = SentimentPredictor()
        if predictor.is_loaded():
            logger.info("✅ Model loaded successfully!")
            model_info = predictor.get_model_info()
            logger.info(f"Model accuracy: {model_info.get('accuracy', 'unknown')}")
        else:
            logger.warning("⚠️ Model not loaded - API will return errors for predictions")
            logger.warning("Please run 'python model/train.py' to train the model first")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {e}")
        predictor = None
    
    logger.info("API initialization complete!")

# Initialize predictor when module loads
initialize_predictor()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers and update metrics"""
    start_time = time.time()
    
    # Update request counter
    request_counter.labels(
        method=request.method, 
        endpoint=request.url.path
    ).inc()
    
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/", response_class=HTMLResponse, tags=["Interface"])
async def root():
    """
    **API Homepage**
    
    Returns a simple HTML page with links to documentation and endpoints.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title> Sentiment Analysis API</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; 
                margin: 40px; 
                background: #f8fafc; 
                color: #334155;
                line-height: 1.6;
            }
            .container { 
                background: white; 
                padding: 40px; 
                border-radius: 12px; 
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                max-width: 800px;
                margin: 0 auto;
            }
            h1 { 
                color: #1e293b; 
                margin-bottom: 8px;
                font-size: 2.5em;
            }
            .subtitle {
                color: #64748b;
                font-size: 1.1em;
                margin-bottom: 30px;
            }
            .links { 
                margin: 30px 0; 
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }
            .links a { 
                display: block;
                padding: 15px 20px; 
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                color: white; 
                text-decoration: none; 
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .links a:hover { 
                transform: translateY(-2px);
                box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
            }
            .status { 
                padding: 15px; 
                margin: 20px 0; 
                border-radius: 8px; 
                border-left: 4px solid;
            }
            .status.healthy { 
                background: #ecfdf5; 
                color: #065f46; 
                border-color: #10b981;
            }
            .status.error { 
                background: #fef2f2; 
                color: #991b1b; 
                border-color: #ef4444;
            }
            .examples {
                background: #f1f5f9;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }
            .examples h3 {
                margin-top: 0;
                color: #475569;
            }
            .example-text {
                background: white;
                padding: 10px;
                border-radius: 4px;
                font-style: italic;
                margin: 5px 0;
                border-left: 3px solid #3b82f6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1> Sentiment Analysis API</h1>
            <p class="subtitle">Production-ready sentiment classification powered by machine learning</p>
            
            <div class="status healthy">
                <strong>✅ API Status:</strong> Online and ready to analyze sentiment!
            </div>
            
            <div class="links">
                <a href="/docs">Interactive Documentation</a>
                <a href="/health">Health Check</a>
                <a href="/metrics">📊 Monitoring Metrics</a>
                <a href="/model/info">Model Information</a>
            </div>
            
            <div class="examples">
                <h3>Try These Examples:</h3>
                <div class="example-text">
                    <strong>Positive:</strong> "This movie is absolutely amazing! Best film I've seen all year."
                </div>
                <div class="example-text">
                    <strong>Negative:</strong> "Terrible film, complete waste of time and money."
                </div>
                <p><strong>💡 Tip:</strong> Visit <a href="/docs" style="color: #3b82f6;">/docs</a> for interactive testing!</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px; color: #64748b;">
                <p>Built with FastAPI • Powered by scikit-learn • IMDB Dataset</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    **Health Check Endpoint**
    
    Returns the current health status of the API and model.
    """
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy" if (predictor and predictor.is_loaded()) else "degraded",
        model_loaded=predictor.is_loaded() if predictor else False,
        version="1.0.0",
        uptime=uptime
    )

@app.post("/predict", response_model=SentimentResponse, tags=["Prediction"])
async def predict_sentiment(request: SentimentRequest):
    """
    **Analyze Sentiment of Text**
    
    Classify the sentiment of the provided text as either positive or negative.
    
    - **text**: Input text to analyze (1-5000 characters)
    - **Returns**: Sentiment prediction with confidence scores
    
    **Example Response:**
    ```json
    {
        "sentiment": "positive",
        "confidence": 0.92,
        "probabilities": {
            "negative": 0.08,
            "positive": 0.92
        }
    }
    ```
    """
    if not predictor or not predictor.is_loaded():
        error_counter.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please contact administrator."
        )
    
    start_time = time.time()
    
    try:
        # Make prediction
        result = predictor.validate_and_predict(request.text)
        
        # Update metrics
        prediction_counter.labels(
            sentiment=result['sentiment'], 
            endpoint="single"
        ).inc()
        prediction_latency.labels(endpoint="single").observe(time.time() - start_time)
        
        logger.info(f"✅ Prediction: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        
        return SentimentResponse(**result)
        
    except ValueError as e:
        error_counter.labels(error_type="validation_error").inc()
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        error_counter.labels(error_type="prediction_error").inc()
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

@app.post("/predict/batch", response_model=BatchSentimentResponse, tags=["Prediction"])
async def predict_batch(request: BatchSentimentRequest):
    """
    **Batch Sentiment Analysis**
    
    Analyze sentiment for multiple texts in a single request.
    
    - **texts**: List of texts to analyze (1-50 texts per request)
    - **Returns**: List of sentiment predictions with processing statistics
    
    **Limits:**
    - Maximum 50 texts per request
    - Each text limited to 5000 characters
    """
    if not predictor or not predictor.is_loaded():
        error_counter.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please contact administrator."
        )
    
    start_time = time.time()
    
    try:
        # Process batch
        raw_results = predictor.predict_batch(request.texts)
        
        # Convert to response models
        results = []
        for result in raw_results:
            results.append(SentimentResponse(**result))
            # Update metrics for each prediction
            prediction_counter.labels(
                sentiment=result['sentiment'], 
                endpoint="batch"
            ).inc()
        
        processing_time = time.time() - start_time
        prediction_latency.labels(endpoint="batch").observe(processing_time)
        
        logger.info(f"✅ Batch processed: {len(results)} texts in {processing_time:.3f}s")
        
        return BatchSentimentResponse(
            results=results,
            total_processed=len(results),
            processing_time=processing_time
        )
        
    except Exception as e:
        error_counter.labels(error_type="batch_error").inc()
        logger.error(f"❌ Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch prediction error")

@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def get_metrics():
    """
    **Prometheus Metrics**
    
    Returns metrics in Prometheus format for monitoring and alerting.
    
    **Available Metrics:**
    - `sentiment_predictions_total`: Total predictions made
    - `sentiment_prediction_duration_seconds`: Prediction latency
    - `sentiment_errors_total`: Total errors
    - `http_requests_total`: Total HTTP requests
    """
    return generate_latest()

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    **Model Information**
    
    Returns detailed information about the trained model.
    """
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    
    try:
        info = predictor.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model information")

# Custom exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return {
        "error": "Endpoint not found", 
        "message": f"The endpoint '{request.url.path}' does not exist",
        "available_endpoints": [
            "/", "/docs", "/health", "/predict", "/predict/batch", "/metrics", "/model/info"
        ]
    }

if __name__ == "__main__":
    # Start Prometheus metrics server on port 8001
    try:
        start_http_server(8001)
        logger.info("📊 Prometheus metrics server started on port 8001")
    except Exception as e:
        logger.warning(f"Could not start Prometheus server: {e}")
    
    # Start FastAPI server
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )