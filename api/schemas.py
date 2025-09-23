from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime

class SentimentRequest(BaseModel):
    """
        Request Schema for sentiment Analysis
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment"
    )

    @field_validator('text')

    @classmethod
    def validate_text(cls, v:str ) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "This movie is absolutely amazing! I loved every second of it."
            }
        }
    }


class SentimentResponse(BaseModel):

    """
        Response schema for sentiment analysis
    """
    sentiment: Literal["positive","negative"] = Field(
        ...,
        description="Confidence score of the prediction (0-1)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,        
        le=1.0,        
        description="Confidence score of the prediction (0-1)"
    )
    probabilities: dict = Field(
        ...,
        description="Preprocessed version of input text (first 100 chars)"
    )
    processed_text: Optional[str] = Field(
        None,
        description="Preprocessed version of input text (first 100 chars)"
    )
    model_version: str  = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "sentiment" : "positive",
                "confidence": 0.92,
                "probabilities": {
                    "negative": 0.08,
                    "positive": 0.92,  
                },
                "processed_text": "movie absolutely amazing loved every second",
                "model_version" : "1.0.0",
                "timestamp" : "2024-01-01T12:00:00"
            }
        }
    }

class BatchSentimentRequest(BaseModel):
    """
        Request for batch sentiment analysis
    """

    texts: List[str] = Field(
        ...,
        description="List of texts to analyze (max 50 per request)"
    )

    @field_validator('texts')

    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:

        if len(v) > 50:
            raise ValueError("Maximum 50 texts allowed per request")
        if len(v) < 1:
            raise ValueError("At least one text is required")
        
        non_empty_texts = [text.strip() for text in v if text.strip()]
        if not non_empty_texts:
            raise ValueError("At least one non-empty text is required")
        return non_empty_texts
    
class BatchSentimentResponse(BaseModel):
    """
        Response for batch sentiment analysis
    """
    results : List[SentimentResponse]
    total_processed: int
    processing_time: float    


class HealthResponse(BaseModel):
    """
        Health check Response
    """
    status: str
    model_loaded: bool
    version: str
    uptime: Optional[float] = None

class ModelInfoResponse(BaseModel):
    """
        Model information response
    """
    model_type: str
    accuracy: float
    training_samples: int
    vocabulary_size: int
    feature_extraction: str
    training_date: str
    sentiment_labels: dict