import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump, load
import nltk
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
def download_nltk_resources():
    resources = ['punkt_tab', 'stopwords', 'wordnet', 'vader_lexicon']
    for resource in resources:
        try:
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Failed to download {resource}: {e}")

# Function to load data from TSV with sampling
def load_data(file_path, sample_size=100000, random_state=42):
    start_time = time.time()
    print(f"Loading data from {file_path}...")
    try:
        # Read with chunksize for large files
        chunks = []
        for chunk in pd.read_csv(file_path, sep='\t', chunksize=50000):
            chunks.append(chunk)
            if sum(len(df) for df in chunks) >= sample_size * 2:
                break
        
        df = pd.concat(chunks)
        # Sample from the loaded data
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=random_state)
        
        print(f"Data loaded and sampled in {time.time() - start_time:.2f} seconds")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Advanced text preprocessing class
class TextPreprocessor:
    def __init__(self):  # Fixed __init__ method name
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vader = SentimentIntensityAnalyzer()
        
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess(self, text):
        """Advanced preprocessing with lemmatization"""
        text = self.clean_text(text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def extract_lexical_features(self, text):
        """Extract sentiment scores using VADER"""
        if not isinstance(text, str) or not text.strip():
            return {"neg": 0, "neu": 0, "pos": 0, "compound": 0}
        return self.vader.polarity_scores(text)

class AdvancedSentimentAnalysisModel:
    def __init__(self):  # Fixed __init__ method name
        self.text_processor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        
    def preprocess_data(self, df):
        start_time = time.time()
        print("Preprocessing data...")
        
        # Verify sentiment column presence
        if 'Sentiment' not in df.columns:
            print("Error: 'Sentiment' column not found in dataset")
            return None, None, None, None
        
        # Copy dataframe to avoid modifying original
        df_processed = df.copy()
        
        # Clean and preprocess text
        print("Cleaning and lemmatizing text...")
        df_processed['cleaned_text'] = df_processed['StudentComments'].apply(self.text_processor.preprocess)
        
        # Extract lexical features
        print("Extracting VADER sentiment scores...")
        vader_scores = df_processed['StudentComments'].apply(self.text_processor.extract_lexical_features)
        
        # Create additional features from VADER
        df_processed['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
        df_processed['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
        df_processed['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
        df_processed['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
        
        # Add text length features
        df_processed['text_length'] = df_processed['StudentComments'].apply(lambda x: len(str(x)))
        df_processed['word_count'] = df_processed['StudentComments'].apply(lambda x: len(str(x).split()))
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df_processed['Sentiment'])
        
        # Features for model
        text_features = df_processed['cleaned_text']
        
        # Additional numerical features
        numerical_features = df_processed[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 
                                           'text_length', 'word_count']]
        
        # Fixed: Use the whole dataset for train/test split
        X_combined = pd.DataFrame({
            'text': text_features,
            'vader_neg': numerical_features['vader_neg'],
            'vader_neu': numerical_features['vader_neu'],
            'vader_pos': numerical_features['vader_pos'],
            'vader_compound': numerical_features['vader_compound'],
            'text_length': numerical_features['text_length'],
            'word_count': numerical_features['word_count']
        })
        
        # Split data - stratify to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Extract text and numerical features back
        X_train_text = X_train['text']
        X_test_text = X_test['text']
        
        X_train_num = X_train[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 
                              'text_length', 'word_count']]
        X_test_num = X_test[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 
                            'text_length', 'word_count']]
        
        # Create combined features dictionary
        X_train_combined = {'text': X_train_text, 'numerical': X_train_num}
        X_test_combined = {'text': X_test_text, 'numerical': X_test_num}
        
        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        print(f"Sentiment classes: {self.label_encoder.classes_}")
        
        return X_train_combined, X_test_combined, y_train, y_test
    
    def build_ensemble_model(self):
        """Build an ensemble model with multiple classifiers"""
        # Base classifiers
        clf1 = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
        clf2 = LinearSVC(dual='auto', C=1.0, max_iter=1000, random_state=42)
        clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('sgd', clf1),
                ('svc', clf2),
                ('rf', clf3)
            ],
            voting='hard'  # Use hard voting as LinearSVC doesn't support predict_proba
        )
        
        return ensemble
    
    def build_pipeline(self):
        """Build pipeline with SMOTE"""
        # Create ensemble classifier
        ensemble = self.build_ensemble_model()
        
        # Create a pipeline with SMOTE for handling class imbalance
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', ensemble)
        ])
        
        return pipeline
    
    def feature_engineering(self, X_train, X_test):
        """Create TF-IDF features from text"""
        start_time = time.time()
        print("Extracting TF-IDF features...")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            min_df=3,
            max_df=0.85,
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            max_features=15000,
            use_idf=True,
            sublinear_tf=True,  # Apply sublinear tf scaling (log)
        )
        
        # Transform text data
        X_train_tfidf = self.vectorizer.fit_transform(X_train['text'])
        X_test_tfidf = self.vectorizer.transform(X_test['text'])
        
        # Convert numerical features to arrays
        X_train_num = X_train['numerical'].values
        X_test_num = X_test['numerical'].values
        
        # Combine TF-IDF and numerical features
        X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_num))
        X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_num))
        
        print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
        print(f"Number of TF-IDF features: {len(self.vectorizer.get_feature_names_out())}")
        print(f"Number of numerical features: {X_train_num.shape[1]}")
        print(f"Total features: {X_train_combined.shape[1]}")
        
        return X_train_combined, X_test_combined
        
    def train(self, X_train, y_train):
        """Train the model with hyperparameter tuning"""
        start_time = time.time()
        print("Training ensemble model with SMOTE...")
        
        # Build the pipeline
        pipeline = self.build_pipeline()
        
        # Train the model
        self.model = pipeline.fit(X_train, y_train)
        
        print(f"Model training completed in {time.time() - start_time:.2f} seconds")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        start_time = time.time()
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        return accuracy, report, conf_matrix
    
    def save_model(self, model_path='advanced_sentiment_model.joblib', 
                   vectorizer_path='advanced_vectorizer.joblib'):
        """Save the trained model and vectorizer to disk"""
        os.makedirs('model', exist_ok=True)
        dump(self.model, f"model/{model_path}")
        dump(self.vectorizer, f"model/{vectorizer_path}")
        dump(self.label_encoder, f"model/advanced_label_encoder.joblib")
        dump(self.text_processor, f"model/text_processor.joblib")
        print(f"Model saved to model/{model_path}")
    
    def load_model(self, model_path='advanced_sentiment_model.joblib',
                  vectorizer_path='advanced_vectorizer.joblib'):
        """Load the trained model and vectorizer from disk"""
        self.model = load(f"model/{model_path}")
        self.vectorizer = load(f"model/{vectorizer_path}")
        self.label_encoder = load(f"model/advanced_label_encoder.joblib")
        self.text_processor = load(f"model/text_processor.joblib")
        print("Model loaded successfully")
    
    def predict_sentiment(self, text, return_probs=False):
        """Predict sentiment for new text"""
        if not isinstance(text, list):
            text = [text]
            
        # Preprocess input text
        processed_text = [self.text_processor.preprocess(t) for t in text]
        
        # Extract lexical features
        vader_features = [self.text_processor.extract_lexical_features(t) for t in text]
        
        # Create DataFrame for numerical features
        numerical_features = pd.DataFrame({
            'vader_neg': [v['neg'] for v in vader_features],
            'vader_neu': [v['neu'] for v in vader_features],
            'vader_pos': [v['pos'] for v in vader_features],
            'vader_compound': [v['compound'] for v in vader_features],
            'text_length': [len(t) for t in text],
            'word_count': [len(t.split()) for t in text]
        })
        
        # Transform text using the vectorizer
        text_tfidf = self.vectorizer.transform(processed_text)
        
        # Combine TF-IDF and numerical features
        features_combined = np.hstack((text_tfidf.toarray(), numerical_features.values))
        
        # Make prediction
        predictions = self.model.predict(features_combined)
        
        # Get the predicted classes
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        
        # Try to get probabilities if available
        try:
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features_combined)
                confidences = np.max(probabilities, axis=1)
            else:
                # For non-probabilistic models like LinearSVC
                confidences = [1.0] * len(predictions)
        except Exception:
            confidences = [1.0] * len(predictions)
        
        results = []
        for i, input_text in enumerate(text):
            result = {
                'text': input_text,
                'sentiment': predicted_classes[i],
                'confidence': confidences[i]
            }
            
            # Include all probabilities if requested
            if return_probs and hasattr(self.model, "predict_proba"):
                probs = {cls: prob for cls, prob in 
                        zip(self.label_encoder.classes_, probabilities[i])}
                result['probabilities'] = probs
                
            results.append(result)
        
        return results[0] if len(results) == 1 else results

def hierarchical_sentiment_predict(model, text):
    """
    Improved hierarchical prediction approach with better neutral case handling
    """
    result = model.predict_sentiment(text, return_probs=True)
    
    # Extract VADER scores directly for this text
    vader_scores = model.text_processor.extract_lexical_features(text)
    
    # Strong neutral signal from VADER (high neutrality and compound near 0)
    if vader_scores['neu'] > 0.60 and abs(vader_scores['compound']) < 0.25:
        result['sentiment'] = 'neutral'
        result['confidence'] = vader_scores['neu']
        return result
    
    # Very strong positive or negative signal from VADER
    if vader_scores['compound'] > 0.5:
        result['sentiment'] = 'positive'
        result['confidence'] = vader_scores['pos']
        return result
    elif vader_scores['compound'] < -0.5:
        result['sentiment'] = 'negative'
        result['confidence'] = vader_scores['neg']
        return result
    
    # If we reach here, use the model probabilities if available
    if hasattr(result, 'probabilities'):
        probs = result['probabilities']
        neutral_prob = probs.get('neutral', 0)
        pos_prob = probs.get('positive', 0)
        neg_prob = probs.get('negative', 0)
        
        # If neutral probability is significant
        if neutral_prob > 0.3:
            result['sentiment'] = 'neutral'
            result['confidence'] = neutral_prob
        # If predicted neutral but stronger signal for other sentiment
        elif result['sentiment'] == 'neutral':
            if pos_prob > 0.4 and pos_prob > neg_prob:
                result['sentiment'] = 'positive'
                result['confidence'] = pos_prob
            elif neg_prob > 0.4 and neg_prob > pos_prob:
                result['sentiment'] = 'negative'
                result['confidence'] = neg_prob
    
    return result

def interactive_mode(model):
    """Interactive terminal mode for sentiment prediction"""
    print("\n" + "="*50)
    print("INTERACTIVE SENTIMENT ANALYSIS MODE")
    print("="*50)
    print("Type a sentence to analyze its sentiment.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("="*50)
    
    while True:
        # Get user input
        user_input = input("\nEnter text to analyze (or 'exit' to quit): ")
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting interactive mode. Goodbye!")
            break
        
        # Skip empty input
        if not user_input.strip():
            print("Please enter some text to analyze.")
            continue
        
        # Predict sentiment using hierarchical approach
        try:
            result = hierarchical_sentiment_predict(model, user_input)
            print(f"\nSentiment: {result['sentiment']} (Confidence: {result['confidence']:.4f})")
            
            # Display VADER scores for additional insight
            vader_scores = model.text_processor.extract_lexical_features(user_input)
            print(f"VADER scores: neg={vader_scores['neg']:.2f}, neu={vader_scores['neu']:.2f}, pos={vader_scores['pos']:.2f}, compound={vader_scores['compound']:.2f}")
            
            # Provide explanation
            explain_prediction(result, vader_scores)
        except Exception as e:
            print(f"Error predicting sentiment: {e}")

def explain_prediction(result, vader_scores):
    """Provide simple explanation of prediction based on VADER scores"""
    sentiment = result['sentiment']
    
    if sentiment == 'positive':
        if vader_scores['pos'] > 0.5:
            print("Explanation: Text contains strongly positive words/phrases.")
        else:
            print("Explanation: Text has more positive than negative elements.")
    elif sentiment == 'negative':
        if vader_scores['neg'] > 0.5:
            print("Explanation: Text contains strongly negative words/phrases.")
        else:
            print("Explanation: Text has more negative than positive elements.")
    else:  # neutral
        if vader_scores['neu'] > 0.8:
            print("Explanation: Text is predominantly factual/objective with few sentiment markers.")
        else:
            print("Explanation: Text contains mixed or balanced sentiment markers.")

def main():
    # Record total execution time
    total_start_time = time.time()
    download_nltk_resources()
    
    # Model file paths
    model_dir = 'model'
    model_path = 'advanced_sentiment_model.joblib'
    vectorizer_path = 'advanced_vectorizer.joblib'
    encoder_path = 'advanced_label_encoder.joblib'
    processor_path = 'text_processor.joblib'
    
    # Check if model already exists
    if (os.path.exists(f"{model_dir}/{model_path}") and 
        os.path.exists(f"{model_dir}/{vectorizer_path}") and 
        os.path.exists(f"{model_dir}/{encoder_path}") and
        os.path.exists(f"{model_dir}/{processor_path}")):
        
        print("Existing model found. Loading model...")
        model = AdvancedSentimentAnalysisModel()
        model.load_model(model_path, vectorizer_path)
    else:
        # Load data
        file_path = 'ReadyToTrain_data_2col_with_subjectivity_final.tsv' 
        df = load_data(file_path, sample_size=10000)
        
        if df is None:
            print("Failed to load dataset. Exiting...")
            return
        
        # Display dataset stats
        print(f"Dataset shape: {df.shape}")
        print("\nSentiment distribution:")
        print(df['Sentiment'].value_counts())
        
        # Create and train model
        model = AdvancedSentimentAnalysisModel()
        
        # Preprocess data
        X_train, X_test, y_train, y_test = model.preprocess_data(df)
        
        if X_train is None:
            print("Preprocessing failed. Exiting...")
            return
        
        # Feature engineering
        X_train_combined, X_test_combined = model.feature_engineering(X_train, X_test)
        
        # Display class distribution
        print(f"\nClass distribution: {np.bincount(y_train)}")
        
        # Train model
        model.train(X_train_combined, y_train)
        
        # Save model for future use
        model.save_model()
        
        # Evaluate model
        accuracy, report, conf_matrix = model.evaluate(X_test_combined, y_test)
    
    # Print total execution time for model training/loading
    model_time = time.time() - total_start_time
    print(f"\nModel preparation time: {model_time:.2f} seconds ({model_time/60:.2f} minutes)")
    
    # Start interactive mode
    interactive_mode(model)

if __name__ == "__main__": 
    main()