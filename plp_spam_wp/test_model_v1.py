import json
import pandas as pd
import pickle
import os
import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 更新NLTK下载的资源，确保所有需要的资源都被下载
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except LookupError:
    print("NLTK资源下载失败，请确保网络连接正常")

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
        f.write("# YouTube Comment Analysis Log (MultinomialNB Model)\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Analysis Process\n\n")
        for msg in LOG_MESSAGES:
            f.write(f"- {msg}\n")
    log_message(f"Log saved to {LOG_FILE}")

def preprocess_text(text):
    """Preprocess text to match training data format"""
    # 确保输入是字符串
    if not isinstance(text, str):
        return ""
    
    try:
        # # 去除HTML标签
        # text = re.sub(r'<.*?>', '', text)
        # # 去除URL
        # text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # # 转换为小写
        # text = text.lower()
        # # 去除非字母字符
        # text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 简单分词，避免使用word_tokenize
        tokens = text.split()
        
        # 去除停用词
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # 词形还原
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        log_message(f"预处理文本时出错: {str(e)}")
        return ""

def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    try:
        # Load model
        with open('spam_model_v1.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load vectorizer
        with open('vectorizer_v1.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        log_message("Model and vectorizer loaded successfully")
        return model, vectorizer
    except FileNotFoundError:
        log_message("Error: Model files not found. Please run train_model_v2.py first to train the model.")
        return None, None
    except Exception as e:
        log_message(f"Error loading model: {str(e)}")
        return None, None

def predict_comment(comment, model, vectorizer):
    """Predict if a comment is spam"""
    if not comment or comment is None or not isinstance(comment, str):
        return "Unable to analyze (empty or non-text comment)"
    
    try:
        # Preprocess comment text to match training data format
        processed_comment = preprocess_text(comment)
        
        # Convert comment text to feature vector
        comment_vector = vectorizer.transform([processed_comment]).toarray()
        
        # Predict result
        prediction = model.predict(comment_vector)
        
        return prediction[0]
    except Exception as e:
        log_message(f"Error predicting comment: {comment[:30]}... - {str(e)}")
        return "Analysis error"

def process_json_file(json_file_path, output_csv_path, model, vectorizer):
    """Process JSON file and save results to CSV"""
    log_message(f"Starting to process file: {json_file_path}")
    
    # Store all comment records
    all_comments = []
    
    try:
        log_message(f"Loading JSON file: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        log_message("JSON file loaded successfully")
        
        total_creators = len(data)
        log_message(f"Found {total_creators} content creators")
        
        # Process each creator
        for creator_idx, (creator, playlists) in enumerate(data.items(), 1):
            log_message(f"Processing creator [{creator_idx}/{total_creators}]: {creator}")
            
            # Process each playlist
            for playlist_id, videos in playlists.items():
                # Process each video
                for video_id, video_data in videos.items():
                    # Extract video information
                    video_title = video_data.get("video_title", "")
                    video_time = video_data.get("video_time", "")
                    
                    # Check if comments array exists
                    if "comments" in video_data and isinstance(video_data["comments"], list):
                        log_message(f"  Processing video: {video_id} - Found {len(video_data['comments'])} comments")
                        
                        # Process each comment
                        for comment_obj in video_data["comments"]:
                            if isinstance(comment_obj, dict) and "Comment" in comment_obj:
                                # Get comment text and timestamp
                                comment_text = comment_obj.get("Comment", "")
                                timestamp = comment_obj.get("Timestamp", "")
                                
                                # Predict if comment is spam
                                prediction = predict_comment(comment_text, model, vectorizer)
                                
                                # Create result record
                                comment_record = {
                                    "Content Creator": creator,
                                    "Playlist ID": playlist_id,
                                    "Video ID": video_id,
                                    "Video Title": video_title,
                                    "Video Time": video_time,
                                    "Timestamp": timestamp,
                                    "Comment": comment_text,
                                    "Prediction": prediction
                                }
                                
                                # Add other fields from comment object
                                for key, value in comment_obj.items():
                                    if key not in ["Comment", "Timestamp"]:
                                        comment_record[key] = value
                                
                                all_comments.append(comment_record)
        
        # Check if any comments were processed
        if not all_comments:
            log_message("Warning: No comment data found")
            save_log()
            return
        
        # Create DataFrame and save to CSV
        log_message(f"Processed {len(all_comments)} comments, saving to CSV...")
        df = pd.DataFrame(all_comments)
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        log_message(f"Results saved to: {output_csv_path}")
        
        # Add summary to log
        log_message("\n## Analysis Summary")
        log_message(f"- Total content creators: {total_creators}")
        log_message(f"- Total comments analyzed: {len(all_comments)}")
        
        # Count predictions
        spam_count = sum(1 for record in all_comments if record["Prediction"] == "SPAM COMMENT")
        non_spam_count = sum(1 for record in all_comments if record["Prediction"] == "NOT A SPAM COMMENT")
        other_count = len(all_comments) - spam_count - non_spam_count
        
        log_message(f"- Spam comments detected: {spam_count}")
        log_message(f"- Non-spam comments detected: {non_spam_count}")
        if other_count > 0:
            log_message(f"- Comments with other results: {other_count}")
        
    except json.JSONDecodeError as e:
        log_message(f"JSON parsing error: {str(e)}")
    except Exception as e:
        log_message(f"Error during processing: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
    
    # Save log file
    save_log()

def main():
    # Check if model files exist
    if not os.path.exists('spam_model_v1.pkl') or not os.path.exists('vectorizer_v1.pkl'):
        log_message("Error: Model files not found. Please run train_model_v1.py first to train the model.")
        save_log()
        return
    
    # Set file paths
    json_file_path = 'data.json'
    output_csv_path = 'comments_analysis_v1.csv'
    
    # Check if JSON file exists
    if not os.path.exists(json_file_path):
        log_message(f"Error: File not found: {json_file_path}")
        save_log()
        return
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    if model is None or vectorizer is None:
        save_log()
        return
    
    # Process JSON file
    process_json_file(json_file_path, output_csv_path, model, vectorizer)

if __name__ == "__main__":
    main() 