import pickle
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import sys
import json

# Add model directory to path
sys.path.append('.')
sys.path.append('./model')

try:
    from model.preprocess import TextPreprocessor
except ImportError:
    logging.warning("Falling back to dummy TextPreprocessor (only lowercasing)")
    # Fallback if import fails
    class TextPreprocessor:
        def preprocess(self, text):
            return text.lower()
        

logger = logging.getLogger(__name__)


class SentimentPredictor:
    def __init__(self, model_path: str = "model/artifacts/"):
        self.model = None
        self.vectorizer = None
        self.preprocessor = None
        self.model_path = Path(model_path)
        self.sentiment_labels = {0: "negative", 1: "positive"}
        self.model_metadata = {}
        self.load_model()

    def load_model(self):
        """
            Load the trained model, vectorizer and the preprocessor
        """

        try:
            # Load model
            with open(self.model_path / 'model.pkl', 'rb') as f:
                self.model = pickle.load(f)

            # Load vectorizer
            with open(self.model_path / 'vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load preprocessor
            with open(self.model_path / 'preprocessor.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # Load metadata
            try:
                with open(self.model_path / 'metadata.json', 'r') as f:
                    self.model_metadata = json.load(f)
            except FileNotFoundError:
                logger.warning("Metadata file not found")
                self.model_metadata = {"version" : "unknown"}

            logger.info("Model, vectorizer and the preprocessor have loaded successfully")
            logger.info(f"Model accuracy: {self.model_metadata.get('accuracy', 'unknown')}")

        except FileNotFoundError:
            logger.error(f"Model files not found: {e}")
            logger.error("Please run 'python model/train.py' first to train the model")
            # set components to None to indicate model not loaded
            self.model        = None
            self.vectorizer   = None
            self.preprocessor = None

        except Exception as e:
            logger.error(f" Error loading model: {e}")
            # set components to None to indicate model not loaded
            self.model        = None
            self.vectorizer   = None
            self.preprocessor = None

    def is_loaded(self) -> bool:
        """
            Check if model is properly loaded
        """
        return all([
            self.model is not None, 
            self.vectorizer is not None, 
            self.preprocessor is not None
        ])
    
    def get_model_info(self) -> Dict:
        """
            Get model metadata and information
        """
        if not self.is_loaded():
            return {
                "error": "Model not loaded",
                "message": "Please train the model first"
            }
        
        return self.model_metadata
    
    def predict_sentiment(self, text: str) -> Dict:
        """
            Predict sentiment for a single text
        """

        if not self.is_loaded():

            logger.error("Model not loaded, cannot make predictions")
            raise RuntimeError("Model not loaded. Please train model first.")
        
        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess(text)

            # Check if processed text is empty
            if not processed_text.strip():
                logger.warning("processed text is emply after preprocessing")
                return {
                    "sentiment": "negative",  # Default fallback
                    "confidence": float(0.5),
                    "probabilities": {
                        "negative": 0.5,
                        "positive": 0.5
                    },
                    "processed_text": "empty_text_after_preprocessing"
                }

            text_vector = self.vectorizer.transform(([processed_text]))
            
            prediction = self.model.predict(text_vector)[0]
            probabilities = self.model.predict_proba(text_vector)[0]


            sentiment = self.sentiment_labels.get(prediction, "negative")

            prob_dist = {}

            for i, prob in enumerate(probabilities):
                label = self.sentiment_labels.get(i, f"class_{i}")
                prob_dist[label] = float(prob)

            confidence = float(max(probabilities))
            
            logger.info(f"Prediction: {sentiment} (confidence: {confidence:.3f})")

            return {
                "sentiment": str(sentiment),
                "confidence": float(confidence),
                "probabilities": prob_dist,
                "processed_text": str(processed_text[:100]) # Truncate for response
            }
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
        
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
            Predict sentiment for multiple texts
        """

        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Please train the model first.")
        
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.predict_sentiment(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting text {i+1}: {e}")
                # Add error result
                results.append({
                    "sentiment": "negative",
                    "confidence": 0.0,
                    "probabilities": {"negative": 0.5, "positive": 0.5},
                    "processed_text": "error_in_processing",
                    "error": str(e)
                })
        
        return results
    
    def validate_and_predict(self, text: str) -> Dict:
        """
            Validate input and predict sentiment
        """
        max_caracter_len = 5000
        # Basic validation
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if len(text) > max_caracter_len:
            raise ValueError(f"Text too long (max {max_caracter_len} characters)")
        
        return self.predict_sentiment(text)