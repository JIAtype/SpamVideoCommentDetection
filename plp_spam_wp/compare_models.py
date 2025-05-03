import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.metrics import confusion_matrix, classification_report

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
        f.write("# Model Comparison Log\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Comparison Process\n\n")
        for msg in LOG_MESSAGES:
            f.write(f"{msg}\n")
    log_message(f"Log saved to {LOG_FILE}")

def load_data():
    """Load training data from CSV"""
    try:
        log_message("Loading training data from Youtube01.csv...")
        df = pd.read_csv('Youtube01.csv')
        
        # Map class labels
        class_mapping = {0: "NOT A SPAM COMMENT", 1: "SPAM COMMENT"}
        df['CLASS'] = df['CLASS'].map(class_mapping)
        
        log_message(f"Total samples: {df.shape[0]}")
        log_message(f"Features: {', '.join(df.columns.tolist())}")
        
        # Count classes
        class_counts = df['CLASS'].value_counts()
        for class_name, count in class_counts.items():
            log_message(f"  - {class_name}: {count} samples")
        
        return df
    except Exception as e:
        log_message(f"Error loading data: {str(e)}")
        return None

def load_model_and_vectorizer(model_version):
    """Load model and vectorizer for the specified version"""
    model_file = f"spam_model_v{model_version}.pkl"
    vectorizer_file = f"vectorizer_v{model_version}.pkl"
    
    try:
        # Check if files exist
        if not os.path.exists(model_file) or not os.path.exists(vectorizer_file):
            log_message(f"Model v{model_version} files not found.")
            return None, None
        
        # Load model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load vectorizer
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        
        log_message(f"Model v{model_version} loaded successfully")
        return model, vectorizer
    except Exception as e:
        log_message(f"Error loading model v{model_version}: {str(e)}")
        return None, None

def evaluate_model(model, vectorizer, data, model_version):
    """Evaluate model on data and return performance metrics"""
    model_name = get_model_name(model)
    log_message(f"\nEvaluating Model v{model_version} ({model_name})...")
    
    try:
        # Prepare data
        X = vectorizer.transform(data['CONTENT']).toarray()
        y_true = data['CLASS']
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        log_message(f"Accuracy: {accuracy:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Display classification report
        report = classification_report(y_true, y_pred)
        log_message("Classification Report:")
        for line in report.split('\n'):
            log_message(line)
        
        return {
            'model_version': model_version,
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred
        }
    except Exception as e:
        log_message(f"Error evaluating model v{model_version}: {str(e)}")
        return None

def get_model_name(model):
    """Get a readable name for the model"""
    class_name = model.__class__.__name__
    model_names = {
        'BernoulliNB': 'Bernoulli Naive Bayes',
        'MultinomialNB': 'Multinomial Naive Bayes',
        'LogisticRegression': 'Logistic Regression',
        'LinearSVC': 'Support Vector Machine (Linear)',
        'RandomForestClassifier': 'Random Forest Classifier'
    }
    return model_names.get(class_name, class_name)

def compare_models(results):
    """Compare multiple models and display/log results"""
    log_message("\n## Model Comparison")
    
    if not results:
        log_message("No valid models to compare.")
        return
    
    # Create comparison table
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    comparison_table = "| Model | " + " | ".join(metrics) + " |\n"
    comparison_table += "|" + "-|" * (len(metrics) + 1) + "\n"
    
    for result in results:
        model_name = f"v{result['model_version']}: {result['model_name']}"
        row = f"| {model_name} | {result['accuracy']:.4f} | {result['precision']:.4f} | {result['recall']:.4f} | {result['f1']:.4f} |"
        comparison_table += row + "\n"
    
    log_message(comparison_table)
    
    # Find best model for each metric
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    best_precision = max(results, key=lambda x: x['precision'])
    best_recall = max(results, key=lambda x: x['recall'])
    best_f1 = max(results, key=lambda x: x['f1'])
    
    log_message("\n## Best Models by Metric")
    log_message(f"- Best Accuracy: v{best_accuracy['model_version']} ({best_accuracy['model_name']}) - {best_accuracy['accuracy']:.4f}")
    log_message(f"- Best Precision: v{best_precision['model_version']} ({best_precision['model_name']}) - {best_precision['precision']:.4f}")
    log_message(f"- Best Recall: v{best_recall['model_version']} ({best_recall['model_name']}) - {best_recall['recall']:.4f}")
    log_message(f"- Best F1-Score: v{best_f1['model_version']} ({best_f1['model_name']}) - {best_f1['f1']:.4f}")
    
    # Provide recommendation
    log_message("\n## Recommendation")
    if best_f1['f1'] > 0.8:
        log_message(f"Recommended model: v{best_f1['model_version']} ({best_f1['model_name']}) with F1-Score of {best_f1['f1']:.4f}")
        log_message("F1-Score provides a good balance between precision and recall.")
    else:
        # If no good F1 score, prioritize based on use case
        log_message("All models have relatively low F1-Scores. Consider which metric is most important for your use case:")
        log_message(f"- If minimizing false positives is critical: v{best_precision['model_version']} ({best_precision['model_name']})")
        log_message(f"- If minimizing false negatives is critical: v{best_recall['model_version']} ({best_recall['model_name']})")
    
    # Plot comparison
    plot_model_comparison(results)

def plot_model_comparison(results):
    """Create and save plots for model comparison"""
    try:
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison', fontsize=16)
        
        # Extract data for plotting
        model_names = [f"v{r['model_version']}" for r in results]
        accuracy = [r['accuracy'] for r in results]
        precision = [r['precision'] for r in results]
        recall = [r['recall'] for r in results]
        f1 = [r['f1'] for r in results]
        
        # Plot accuracy
        axs[0, 0].bar(model_names, accuracy)
        axs[0, 0].set_title('Accuracy')
        axs[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracy):
            axs[0, 0].text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        # Plot precision
        axs[0, 1].bar(model_names, precision)
        axs[0, 1].set_title('Precision')
        axs[0, 1].set_ylim(0, 1)
        for i, v in enumerate(precision):
            axs[0, 1].text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        # Plot recall
        axs[1, 0].bar(model_names, recall)
        axs[1, 0].set_title('Recall')
        axs[1, 0].set_ylim(0, 1)
        for i, v in enumerate(recall):
            axs[1, 0].text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        # Plot F1-score
        axs[1, 1].bar(model_names, f1)
        axs[1, 1].set_title('F1-Score')
        axs[1, 1].set_ylim(0, 1)
        for i, v in enumerate(f1):
            axs[1, 1].text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_file = "model_comparison_plot.png"
        plt.savefig(plot_file)
        log_message(f"\nComparison plot saved to {plot_file}")
        plt.close()
        
        # Add the plot to MD file
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n\n![Model Comparison]({plot_file})\n")
        
    except Exception as e:
        log_message(f"Error creating comparison plot: {str(e)}")

def main():
    # Load data
    data = load_data()
    if data is None:
        save_log()
        return
    
    # Define model versions to evaluate
    model_versions = [1, 2, 3, 4, 5, 6, 7, 8]  # Models v1-v8
    
    # Evaluate each model
    results = []
    for version in model_versions:
        model, vectorizer = load_model_and_vectorizer(version)
        if model is not None and vectorizer is not None:
            result = evaluate_model(model, vectorizer, data, version)
            if result is not None:
                results.append(result)
    
    # Compare models
    if results:
        compare_models(results)
    else:
        log_message("No valid models to compare.")
    
    # Save log
    save_log()

if __name__ == "__main__":
    main() 