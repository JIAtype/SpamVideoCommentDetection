# YouTube Spam Comment Detection System

A machine learning system to detect spam comments in YouTube videos using different Naive Bayes classifier models.

## System Components

This system consists of several components:

1. **Model Training** - Train different Naive Bayes models on YouTube comment data
   - `train_model_v1.py` - Trains a BernoulliNB model
   - `train_model_v2.py` - Trains a MultinomialNB model

2. **Model Testing** - Test trained models on YouTube comments
   - `test_model_v1.py` - Tests the BernoulliNB model on JSON data
   - `test_model_v2.py` - Tests the MultinomialNB model on JSON data
   - `interactive_test.py` - Interactive CLI for testing individual comments

3. **Model Comparison** - Evaluate and compare model performance
   - `compare_models.py` - Directly compares both models on accuracy and other metrics

## Data Format

The system works with two types of data:

1. **Training Data** - `Youtube01.csv` with the following columns:
   - `CONTENT`: The text content of the comment
   - `CLASS`: 0 for not spam, 1 for spam

2. **Test Data (JSON)** - `data.json` with the following structure:
   ```json
   {
     "Content Creator": {
       "Playlist ID": {
         "Video ID": {
           "video_title": "",
           "video_time": "",
           "comments": [
             {
               "Timestamp": "2024-07-22T20:50:29Z",
               "Comment": "Comment text"
             }
           ]
         }
       }
     }
   }
   ```

## Usage

### 1. Training Models

Train the BernoulliNB model:
```bash
python train_model_v1.py
```

Train the MultinomialNB model:
```bash
python train_model_v2.py
```

These scripts will:
- Load and prepare the data from `Youtube01.csv`
- Perform 5-fold cross-validation
- Train and evaluate the model
- Save the trained model and vectorizer
- Generate a detailed log file with training metrics

### 2. Testing Models on JSON Data

Test the BernoulliNB model:
```bash
python test_model_v1.py
```

Test the MultinomialNB model:
```bash
python test_model_v2.py
```

These scripts will:
- Load the trained model and vectorizer
- Process comments from the `data.json` file
- Predict whether each comment is spam or not
- Save results to a CSV file with predictions
- Generate a detailed log file with testing metrics

### 3. Interactive Testing

Test either model interactively:
```bash
python interactive_test.py --model 1  # For BernoulliNB (v1)
```
or
```bash
python interactive_test.py --model 2  # For MultinomialNB (v2)
```

This script will:
- Let you input comments directly
- Show real-time prediction results
- Record all interactions in a log file

### 4. Comparing Models

Compare both models' performance:
```bash
python compare_models.py
```

This script will:
- Evaluate both models on the same dataset
- Generate detailed performance metrics
- Create a side-by-side comparison
- Provide a recommendation for which model is better
- Save all results to a log file

## Log Files

All scripts automatically generate log files named after the script and execution time:
```
[script_name]_[YYYYMMDD_HHMMSS].md
```

These logs provide detailed information about the execution process and results in Markdown format, making them easy to read and share.

## Model Selection

The `compare_models.py` script will help you determine which model performs better for your specific use case. In general:

- **BernoulliNB** often works well when features are binary (presence/absence)
- **MultinomialNB** typically works well for text classification with word counts

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
