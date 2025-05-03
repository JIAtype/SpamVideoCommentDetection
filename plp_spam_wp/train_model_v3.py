import pandas as pd
import numpy as np
import pickle
import datetime
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Update NLTK resource downloads to ensure all required resources are available
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except LookupError:
    print("NLTK resource download failed, please check your internet connection")

# Initialize log file
def get_log_filename():
    """Generate log filename based on script name and execution time"""
    script_name = os.path.basename(__file__).split('.')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{script_name}_{timestamp}.md"

# Global variables for logging
LOG_MESSAGES = []
LOG_FILE = get_log_filename()

def log_message(message):
    """Log a message to both console and log list"""
    print(message)
    LOG_MESSAGES.append(message)

def save_log():
    """Save all logged messages to the log file"""
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("# YouTube Spam Detection Model Training Log (LogisticRegression)\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Training Process\n\n")
        for msg in LOG_MESSAGES:
            f.write(f"- {msg}\n")
    log_message(f"Log saved to {LOG_FILE}")

# Load data
log_message("Loading data from Youtube01.csv...")
data = pd.read_csv("Youtube01.csv") 
# We only need content and class columns
data = data[['CONTENT','CLASS']]

# Map 0 to not spam and 1 to spam
data["CLASS"] = data['CLASS'].map({0:'NOT A SPAM COMMENT', 1: 'SPAM COMMENT'})
log_message("Data preparation completed")

def preprocess_text(text):
    """Preprocess text, remove HTML tags, URLs, punctuation and perform lemmatization"""
    # Ensure input is a string
    if not isinstance(text, str):
        return ""
    
    try:
        # # Remove HTML tags
        # text = re.sub(r'<.*?>', '', text)
        # # Remove URLs
        # text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # # Convert to lowercase
        # text = text.lower()
        # # Remove non-alphabetic characters
        # text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Simple tokenization, avoid using word_tokenize
        tokens = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        log_message(f"Error preprocessing text: {str(e)}")
        return ""

log_message("Starting text preprocessing...")
data['processed_content'] = data['CONTENT'].apply(preprocess_text)
log_message(f"Text preprocessing completed, processed {len(data)} comments")

# Prepare features and labels
x = np.array(data['processed_content'])
y = np.array(data['CLASS'])

# Create text feature vectors
log_message("Creating text feature vectors...")
cv = CountVectorizer()
x = cv.fit_transform(x)
log_message(f"Feature vector created with {x.shape[1]} features")

# Split training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.2, random_state=42)
log_message(f"Data split into training ({xtrain.shape[0]} samples) and test ({xtest.shape[0]} samples) sets")

# Train model
log_message("Training LogisticRegression model...")
model = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
model.fit(xtrain, ytrain)

# Cross-validation
log_message("Performing 5-fold cross-validation...")
cv_scores = cross_val_score(LogisticRegression(max_iter=1000, C=1.0, solver='liblinear'), x, y, cv=5)
log_message(f"Cross-validation scores: {', '.join([f'{score:.4f}' for score in cv_scores])}")
log_message(f"Mean CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Evaluate model
accuracy = model.score(xtest, ytest)
log_message(f"Model accuracy on test set: {accuracy:.4f}")

# Detailed classification report
y_pred = model.predict(xtest)
log_message("\n## Classification Report")
report = classification_report(ytest, y_pred, output_dict=True)
log_message(f"Precision (Spam): {report['SPAM COMMENT']['precision']:.4f}")
log_message(f"Recall (Spam): {report['SPAM COMMENT']['recall']:.4f}")
log_message(f"F1-score (Spam): {report['SPAM COMMENT']['f1-score']:.4f}")
log_message(f"Precision (Not Spam): {report['NOT A SPAM COMMENT']['precision']:.4f}")
log_message(f"Recall (Not Spam): {report['NOT A SPAM COMMENT']['recall']:.4f}")
log_message(f"F1-score (Not Spam): {report['NOT A SPAM COMMENT']['f1-score']:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(ytest, y_pred)
log_message("\n## Confusion Matrix")
log_message(f"True Negatives: {conf_matrix[0][0]}")
log_message(f"False Positives: {conf_matrix[0][1]}")
log_message(f"False Negatives: {conf_matrix[1][0]}")
log_message(f"True Positives: {conf_matrix[1][1]}")

# Save model and vectorizer
log_message("Saving model and vectorizer...")
with open('spam_model_v3.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('vectorizer_v3.pkl', 'wb') as f:
    pickle.dump(cv, f)

log_message("Model and vectorizer saved to 'spam_model_v3.pkl' and 'vectorizer_v3.pkl'")

# Save log file
save_log() 