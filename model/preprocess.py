import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:
        
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'<.*?>','', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = text.translate(str.maketrans('','', string.punctuation))
        text = re.sub(r'\d+', '', text)

        tokens = text.split()

        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) >2
        ]
        
        return ' '.join(tokens)
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
            Preprocess multile texts
        """
        return [self.preprocess(text) for text in texts]