import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
from sklearn.utils import resample
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
# 文本预处理函数
def preprocess_text(text):
    if isinstance(text, str):
        # 转为小写
        text = text.lower()
        # 删除特殊字符、数字和标点
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # 分词
        words = nltk.word_tokenize(text)
        # 去除停用词
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        # 词干提取
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        # 重新组合成文本
        text = ' '.join(words)
        return text
    return ''

# 加载数据
print("加载和准备数据...")
data = pd.read_csv("Youtube01.csv")
# 只需要内容和类别列
data = data[['CONTENT', 'CLASS']]

# 检查缺失值
print(f"缺失值统计：\n{data.isnull().sum()}")

# 类别分布
print(f"类别分布：\n{data['CLASS'].value_counts()}")

# 将0映射为非垃圾评论，1映射为垃圾评论
data["CLASS"] = data['CLASS'].map({0: 'NOT_SPAM', 1: 'SPAM'})

# 应用文本预处理
print("正在预处理文本...")
data['PROCESSED_CONTENT'] = data['CONTENT'].apply(preprocess_text)

# 处理类别不平衡（如果存在）
spam = data[data['CLASS'] == 'SPAM']
not_spam = data[data['CLASS'] == 'NOT_SPAM']

# 检查是否存在类别不平衡
if len(spam) < len(not_spam):
    print(f"检测到类别不平衡: SPAM({len(spam)}) vs NOT_SPAM({len(not_spam)})")
    # 对少数类进行上采样
    spam_upsampled = resample(spam, 
                              replace=True,
                              n_samples=len(not_spam),
                              random_state=42)
    # 合并上采样的数据
    balanced_data = pd.concat([not_spam, spam_upsampled])
    print(f"上采样后的类别分布: {balanced_data['CLASS'].value_counts()}")
    data = balanced_data

# 准备特征和标签
X = np.array(data['PROCESSED_CONTENT'])
y = np.array(data['CLASS'])

# 分割训练集和测试集（注意：使用80%数据进行训练）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

# 创建模型评估函数
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(report)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('预测标签')
    plt.ylabel('实际标签')
    plt.title('混淆矩阵')
    plt.show()
    
    # 如果模型支持概率预测，绘制PR曲线
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)
        # 假设第二列是正类的概率
        spam_probas = probas[:, list(model.classes_).index('SPAM')]
        
        precision, recall, _ = precision_recall_curve(
            [1 if label == 'SPAM' else 0 for label in y_test], 
            spam_probas
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.grid(True)
        plt.show()
    
    return accuracy, report

# 使用TF-IDF向量化，并尝试不同的分类模型
print("正在构建和评估模型...")

# 创建TF-IDF向量化器
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# 1. 逻辑回归模型
print("\n评估逻辑回归模型...")
lr_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'))
])
lr_pipeline.fit(X_train, y_train)
lr_accuracy, lr_report = evaluate_model(lr_pipeline, X_test, y_test)

# 2. SVM模型
print("\n评估SVM模型...")
svm_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('scaler', StandardScaler(with_mean=False)),  # TF-IDF特征通常是稀疏的
    ('classifier', SVC(kernel='linear', probability=True, class_weight='balanced'))
])
svm_pipeline.fit(X_train, y_train)
svm_accuracy, svm_report = evaluate_model(svm_pipeline, X_test, y_test)

# 3. 随机森林模型
print("\n评估随机森林模型...")
rf_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced'))
])
rf_pipeline.fit(X_train, y_train)
rf_accuracy, rf_report = evaluate_model(rf_pipeline, X_test, y_test)

# 比较模型性能并选择最佳模型
print("\n模型性能比较：")
models = {
    "逻辑回归": (lr_pipeline, lr_accuracy),
    "SVM": (svm_pipeline, svm_accuracy),
    "随机森林": (rf_pipeline, rf_accuracy)
}

best_model_name = max(models.items(), key=lambda x: x[1][1])[0]
best_model, best_accuracy = models[best_model_name]

print(f"最佳模型是: {best_model_name}，准确率: {best_accuracy:.4f}")

# 使用交叉验证进一步评估最佳模型
print("\n对最佳模型进行交叉验证...")
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"交叉验证平均准确率: {np.mean(cv_scores):.4f}, 标准差: {np.std(cv_scores):.4f}")

# 保存最佳模型
print("保存最佳模型...")
with open('best_spam_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("模型已保存到 'spam_model.pkl'")

# 创建一个简单的预测函数
def predict_spam(text, model):
    # 预处理文本
    processed_text = preprocess_text(text)
    # 使用模型预测
    prediction = model.predict([processed_text])[0]
    # 获取概率
    probabilities = model.predict_proba([processed_text])[0]
    # 找到SPAM类别的索引
    spam_index = list(model.classes_).index('SPAM')
    spam_prob = probabilities[spam_index]
    
    return prediction, spam_prob

# 示例预测
sample_comments = [
    "Great video! I learned a lot from this.",
    "Check out my channel for free subscribers and views!",
    "This is the best tutorial I've seen on this topic.",
    "Free iPhone giveaway! Click this link to participate: http://bit.ly/scam"
]

print("\n示例预测:")
for comment in sample_comments:
    pred, prob = predict_spam(comment, best_model)
    print(f"评论: '{comment}'")
    print(f"预测: {pred}, 垃圾评论概率: {prob:.4f}\n")