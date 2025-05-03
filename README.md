# Spam Comments Detection Model Details

Analyze the performance of five machine learning models on the task of spam comment detection and record the relevant data in detail, each model is tested under two data processing scenarios: **raw data (labeled “ wp”)** and **pre-processed data**.

## Key Findings

* Impact of preprocessing: All models outperform the raw, unprepared data by 5-15%.
* Best models: SVM and **RandomForest** outperform on raw data, both achieving 93.61% accuracy in testing
* Stability: **RandomForest** has the lowest fluctuation (±0.0132) in cross-validation, **indicating** that it has the best stability.
* Loss after preprocessing: **RandomForest** suffers the worst performance drop after preprocessing, from 93.61% to 80.7

## Detailed Model Analysis

### SVM (**LinearSVC**)

* Strengths: Best performance on the test set (tied with **RandomForest**), highest F1 score (0.9364)
* Balance: Spam and non-spam comments have **almost the** same F1 score (0.9364 vs. 0.9358), **indicating** a very balanced categorization
* Robustness: The performance on preprocessed data is still relatively good, with an accuracy of 85.05%.

### RandomForest

* Strengths: high test accuracy (93.61%), best model stability (±0.0132)
* Spam accuracy: 98.65% on raw data, highest accuracy rate
* Disadvantages: Extremely sensitive to preprocessing, accuracy drops by 13 percentage points after preprocessing

### LogisticRegression

* Strengths: Overall excellent performance, 93.2% accuracy on cross-validation, 92.78% accuracy on testing
* Adaptability: 86.65% accuracy on preprocessed data, adaptability is strong
* Characteristics: High recall rate of non-spam comments (98.38%), suitable for ensuring that **important information** is not misclassified as spam.

### MultinomialNB

* Strengths: Highest spam recall (96.23%), good for detecting all potentially spammy comments as much as possible
* Characteristics: Minimal difference in performance before and after preprocessing among all models, **relatively stable**
* Disadvantage: Overall accuracy is lower than the **previous** three models

### BernoulliNB

* Strengths: **Very high** recall rate for non-spam comments (99.33%)
* Disadvantage: Lowest overall accuracy, less than 70% recall of spam comments
* Characteristics: Cross-validation fluctuates the most (±0.0832), model stability is the worst

## Practical Application Analysis

### Differences in filtering aggressiveness:

* **RandomForest** detected the most spam comments (34,566) on the preprocessed **data, but** may **contain** **a large number of** false positives.
* **LogisticRegression** was the most conservative on the original data, marking only 6,368 spam comments.
* **MultinomialNB** detected 33,061 spam comments on the original data and was the most aggressive filter.

### Model selection:

* High precision requirement: SVM or **LogisticRegression** (original data) are best suited for scenarios that require high accuracy.
* High recall requirement: **MultinomialNB** (original data) is suitable for scenarios that need to capture as many spam comments as possible.
* Balanced requirement: **RandomForest** (original data) provides the most balanced F1 score, with both spam and non-spam categories at 0.9361.
