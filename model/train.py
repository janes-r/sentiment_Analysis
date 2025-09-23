import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from model.preprocess import TextPreprocessor
import logging

#Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def load_imdb_data():
    """
    IMBD dataset from kaggle
    """

    try:
        df = pd.read_csv('data/IMDB_dataset.csv')
        logger.info(f"Loaded {len(df)} reviews from local dataset")
        return df

    except FileNotFoundError:
        logger.warning("IMDB_dataset.csv dataset was not found in the repertory data")
        return None
    
def train_sentiment_model():
    """
    Train sentiment classification model
    """

    max_features = 5000

    df = load_imdb_data()

    # Check data distribution
    sentiment_counts = df['sentiment'].value_counts()
    logger.info(f"Data distribution:\n{sentiment_counts}")

    #sentiment_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    sentiment_map = {'positive': 1, 'negative': 0}
    # Handle IMBD dataset
    df['label'] = df['sentiment'].map(sentiment_map)
    '''
    if 'sentiment' in df.columns:
        df['label'] = df['sentiment'].map(sentiment_map)
    else:
        df['label'] = df['sentiment'].map({'positive': 2,'negative': 0})
    '''
    # Drop any rows with missing labels
    df = df.dropna(subset=['label'])

    # Preprocess text
    logger.info(" Preprocessing text data...")

    preprocessor = TextPreprocessor()
    df['processed_text'] = df['review'].apply(preprocessor.preprocess)

    # Remove empty processed texts
    df = df[df['processed_text'].str.len() > 0]
    logger.info(f"Final dataset size: {len(df)} reviews")

    X = df['processed_text'].values
    Y = df['label'].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    ) 

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Testing set size: {X_test}")


    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    logger.info(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    logger.info(f"Training matrix shape: {X_train_vec.shape}")

    # Train model
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train_vec, Y_train)

    logger.info("Model training completed!")

    # Evaluate the model

    logger.info("Evaluating model performance...")

    Y_pred = model.predict(X_test_vec)
    Y_pred_proba = model.predict_proba(X_test_vec)

    accuracy = accuracy_score(Y_test, Y_pred)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")

    # Create label names for report
    label_names = ['negative', 'neutral', 'positive']
    unique_labels = sorted(np.unique(Y_test))
    report_labels = [label_names[i] for i in unique_labels]

    print(classification_report(Y_test, Y_pred, target_names=report_labels))

    cm = confusion_matrix(Y_test, Y_pred)
    print(f"\nConfusion Matrix:")
    print(f"           Predicted")
    print(f"         Neg    Pos")
    print(f"Actual Neg {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Pos {cm[1,0]:4d}  {cm[1,1]:4d}")


    print(f"\nSample Predictions:")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for i, idx in enumerate(sample_indices):
        text = X_test[idx][:100] + "..." if len(X_test[idx]) > 100 else X_test[idx]
        actual = "positive" if Y_test[idx] == 1 else "negative"
        predicted = "positive" if Y_pred[idx] == 1 else "negative"
        confidence = max(Y_pred_proba[idx])
        
        print(f"{i+1}. Text: '{text}'")
        print(f"   Actual: {actual}, Predicted: {predicted}, Confidence: {confidence:.3f}")
        print()


    # Save artifacts

    Path('model/artifacts').mkdir(parents=True, exist_ok=True)

    with open('model/artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open("model/artifacts/vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)

    with open('model/artifacts/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    logger.info("Model and vectorizer saved to models/artifacts/")

    nb_top_features = 100

    metadata = {
        'model_type' : 'LogisticRegression',
        'accuracy'   : float(accuracy),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'vocabulary_size': len(vectorizer.get_feature_names_out()),
        'feature_extraction': 'TF-IDF',
        'ngram_range': [1, 2],
        'max_features': max_features,
        'sentiment_labels': {0: 'negative', 1: 'positive'},
        'features'   : vectorizer.get_feature_names_out()[:nb_top_features].tolist(),
        'model_type' : 'LogisticRegression',
        'n_classes'  : len(np.unique(Y_train)),
        'training_date': pd.Timestamp.now().isoformat()
    }

    import json

    with open('model/artifacts/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


    return model, vectorizer, preprocessor, accuracy

def test_model():
    """Test the trained model with sample texts"""
    logger.info(" Testing trained model...")
    
    # Load artifacts
    artifacts_dir = Path('model/artifacts')
    
    try:
        with open(artifacts_dir / 'model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(artifacts_dir / 'vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open(artifacts_dir / 'preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except FileNotFoundError:
        logger.error("Model artifacts not found. Please run training first.")
        return
    
    # Test cases
    test_texts = [
        "This movie is absolutely amazing! I loved every second of it.",
        "Terrible film. Waste of time and money.",
        "The acting was good but the plot was confusing.",
        "Best movie I've seen this year! Highly recommended.",
        "Boring and predictable storyline."
    ]
    
    print("\nModel Testing Results:")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):

        processed = preprocessor.preprocess(text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = max(probability)
        
        print(f"{i}. Text: '{text}'")
        print(f"   Prediction: {sentiment} (confidence: {confidence:.3f})")
        print(f"   Probabilities: negative={probability[0]:.3f}, positive={probability[1]:.3f}")
        print()



if __name__ == "__main__":

    # Train the model

    model, vectorizer, preprocessor, accuracy =  train_sentiment_model()
   
    print(f"\n Final Results:")
    print(f"\n Training complete! Accuracy {accuracy: .2%}")
    print(f"   Model Type: Logistic Regression")
    print(f"   Feature Extraction: TF-IDF")


    # Test the model

    test_model()
    
    print(f"\n Training complete! Model ready for deployment.")
    print(f" Artifacts saved in: model/artifacts/")
    print(f" Next step: Run the API server!")