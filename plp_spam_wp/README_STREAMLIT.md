# YouTube评论垃圾检测系统 - Streamlit应用

这个应用使用机器学习模型来检测YouTube评论中的垃圾内容，提供了一个友好的Web界面来进行交互式使用。

## 功能特点

- 支持多个训练好的模型版本（v1-v5）
- 实时评论分析和结果展示
- 历史记录和统计信息展示
- 数据可视化（饼图显示垃圾评论与正常评论比例）
- 用户友好的界面设计

## 安装说明

1. 确保已安装Python 3.8+

2. 安装所需的依赖:
```bash
pip install -r requirements.txt
```

## 运行应用

执行以下命令启动Streamlit应用:
```bash
streamlit run app.py
```

应用将自动在您的浏览器中打开（通常是http://localhost:8501）。

## 使用方法

1. 在左侧边栏中选择模型版本（默认为模型v1）
2. 在文本框中输入要分析的YouTube评论
3. 点击"分析评论"按钮
4. 查看分析结果和统计信息
5. 使用标签页切换不同视图：历史记录、统计分析、使用说明

## 注意事项

- 如果出现模型版本不匹配的警告，请确保安装的scikit-learn版本与模型训练时使用的版本匹配（推荐使用scikit-learn 1.5.2）
- 结果仅供参考，最终判断应结合具体情况
- 该应用是一个辅助工具，不能100%准确地检测所有垃圾评论

## 技术栈

- Streamlit：Web界面框架
- Scikit-learn：机器学习库
- Pandas：数据处理
- Matplotlib：数据可视化

## 文件说明

- `app.py`: Streamlit应用主文件
- `spam_model_v*.pkl`: 训练好的模型文件
- `vectorizer_v*.pkl`: 对应的特征向量化器文件
- `requirements.txt`: 依赖项列表 