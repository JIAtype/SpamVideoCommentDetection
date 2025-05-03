import pickle
import datetime
import os
import argparse

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
        f.write("# YouTube Comment Interactive Testing Log\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Testing Process\n\n")
        for msg in LOG_MESSAGES:
            f.write(f"- {msg}\n")
    log_message(f"Log saved to {LOG_FILE}")

def load_model_and_vectorizer(model_version):
    """Load the trained model and vectorizer"""
    model_file = f'spam_model_v{model_version}.pkl'
    vectorizer_file = f'vectorizer_v{model_version}.pkl'
    
    try:
        # Load model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load vectorizer
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        
        model_type = "BernoulliNB" if model_version == 1 else "MultinomialNB"
        log_message(f"Model ({model_type}) and vectorizer loaded successfully")
        return model, vectorizer
    except FileNotFoundError:
        log_message(f"Error: Model files not found. Please run train_model_v{model_version}.py first to train the model.")
        return None, None
    except Exception as e:
        log_message(f"Error loading model: {str(e)}")
        return None, None

def predict_comment(comment, model, vectorizer):
    """Predict if a comment is spam"""
    if not comment or not isinstance(comment, str) or comment.strip() == "":
        return "Unable to analyze (empty comment)"
    
    try:
        # Convert comment text to feature vector
        comment_vector = vectorizer.transform([comment]).toarray()
        
        # Predict result
        prediction = model.predict(comment_vector)
        
        return prediction[0]
    except Exception as e:
        log_message(f"Error predicting comment: {str(e)}")
        return "Analysis error"

def run_interactive_test(model_version):
    """Run interactive test with the specified model version"""
    log_message(f"Starting interactive test with model version {model_version}")
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer(model_version)
    if model is None or vectorizer is None:
        save_log()
        return
    
    log_message("Interactive testing started. Enter a comment to analyze or 'q' to quit.")
    print("\n" + "="*60)
    print(f"YouTube Comment Spam Detector (Model v{model_version})")
    print("Enter a comment to analyze or 'q' to quit")
    print("="*60 + "\n")
    
    test_count = 0
    spam_count = 0
    not_spam_count = 0
    
    while True:
        # Get user input
        user_input = input("\nEnter comment: ")
        
        # Check if user wants to quit
        if user_input.lower() == 'q':
            log_message("User ended interactive testing")
            break
        
        # Predict if comment is spam
        result = predict_comment(user_input, model, vectorizer)
        test_count += 1
        
        # Display result
        if result == "SPAM COMMENT":
            print("\n🚨 Result: SPAM 🚨")
            spam_count += 1
        elif result == "NOT A SPAM COMMENT":
            print("\n✓ Result: NOT SPAM")
            not_spam_count += 1
        else:
            print(f"\nResult: {result}")
        
        # Log the result
        log_message(f"Test {test_count} - Comment: \"{user_input}\" - Result: {result}")
    
    # Display summary
    print("\n" + "="*60)
    print(f"Testing Summary:")
    print(f"Total comments tested: {test_count}")
    print(f"Spam comments detected: {spam_count}")
    print(f"Non-spam comments detected: {not_spam_count}")
    print("="*60 + "\n")
    
    # Log summary
    log_message("\n## Testing Summary")
    log_message(f"- Total comments tested: {test_count}")
    log_message(f"- Spam comments detected: {spam_count}")
    log_message(f"- Non-spam comments detected: {not_spam_count}")
    
    # Save log
    save_log()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='YouTube Comment Spam Detector - Interactive Testing')
    parser.add_argument('--model', type=int, choices=[1, 2], default=2,
                        help='Model version to use: 1 for BernoulliNB, 2 for MultinomialNB (default: 2)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run interactive test with specified model
    run_interactive_test(args.model)

if __name__ == "__main__":
    main() 