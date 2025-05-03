import pandas as pd
import numpy as np
import os
import random

# 设置显示选项
pd.set_option('display.max_colwidth', 100)

def main():
    print("YouTube评论垃圾检测结果分析")
    print("=" * 50)
    
    # 指定CSV文件路径
    file_path = 'comments_analysis_v1.csv'
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return
    
    # 加载数据
    try:
        print(f"正在加载数据: {file_path}")
        df = pd.read_csv(file_path)
        print(f"成功加载数据，共 {df.shape[0]} 条评论")
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return
    
    # 显示数据统计信息
    print("\n数据统计信息:")
    print("-" * 50)
    
    # 检查预测列是否存在
    if 'Prediction' not in df.columns:
        print("错误: 数据中没有'Prediction'列")
        print(f"可用的列: {', '.join(df.columns)}")
        return
    
    # 统计预测结果
    prediction_counts = df['Prediction'].value_counts()
    print("\n预测结果统计:")
    for label, count in prediction_counts.items():
        print(f"  - {label}: {count} ({count/len(df)*100:.2f}%)")
    
    # 筛选垃圾评论和非垃圾评论
    spam_comments = df[df['Prediction'] == 'SPAM COMMENT']
    non_spam_comments = df[df['Prediction'] == 'NOT A SPAM COMMENT']
    
    print(f"\n找到 {len(spam_comments)} 条垃圾评论")
    print(f"找到 {len(non_spam_comments)} 条非垃圾评论")
    
    # 随机选择10条垃圾评论和10条非垃圾评论
    random_spam = spam_comments.sample(min(10, len(spam_comments)))
    random_non_spam = non_spam_comments.sample(min(10, len(non_spam_comments)))
    
    # 显示随机垃圾评论
    print("\n\n=== 随机10条垃圾评论 ===\n")
    for i, (_, row) in enumerate(random_spam.iterrows(), 1):
        print(f"垃圾评论 #{i}:")
        print(f"创作者: {row.get('Content Creator', 'N/A')}")
        if 'Video Title' in row:
            print(f"视频标题: {row['Video Title']}")
        print(f"评论内容: {row['Comment']}")
        print("-" * 80)
    
    # 显示随机非垃圾评论
    print("\n\n=== 随机10条非垃圾评论 ===\n")
    for i, (_, row) in enumerate(random_non_spam.iterrows(), 1):
        print(f"非垃圾评论 #{i}:")
        print(f"创作者: {row.get('Content Creator', 'N/A')}")
        if 'Video Title' in row:
            print(f"视频标题: {row['Video Title']}")
        print(f"评论内容: {row['Comment']}")
        print("-" * 80)
    
    # 保存随机样本到CSV文件
    random_samples = pd.concat([random_spam, random_non_spam])
    output_file = 'random_comment_samples.csv'
    random_samples.to_csv(output_file, index=False)
    print(f"\n随机评论样本已保存到: {output_file}")

if __name__ == "__main__":
    main() 