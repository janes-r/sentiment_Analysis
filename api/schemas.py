from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datatime import datatime

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
        return v.strip
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "This movie is absolutely amazing! I loved every second of it."
            }
        }
    }


