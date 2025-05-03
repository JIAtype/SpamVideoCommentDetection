import streamlit as st
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

# 页面配置
st.set_page_config(
    page_title="YouTube评论垃圾检测系统",
    page_icon="🚨",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .spam-result {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
    }
    .spam {
        background-color: #ffcccc;
        color: #cc0000;
    }
    .not-spam {
        background-color: #ccffcc;
        color: #008800;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#4e54c8, #8f94fb);
    }
    h1, h2 {
        color: #4e54c8;
    }
</style>
""", unsafe_allow_html=True)

# 加载模型和向量化器
@st.cache_resource
def load_model_and_vectorizer(model_version):
    """加载训练好的模型和向量化器"""
    model_file = f'spam_model_v{model_version}.pkl'
    vectorizer_file = f'vectorizer_v{model_version}.pkl'
    
    try:
        # 加载模型
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # 加载向量化器
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer, True
    except Exception as e:
        return None, None, str(e)

# 预测评论
def predict_comment(comment, model, vectorizer):
    """预测评论是否为垃圾评论"""
    if not comment or not isinstance(comment, str) or comment.strip() == "":
        return "无法分析（空评论）"
    
    try:
        # 将评论文本转换为特征向量
        comment_vector = vectorizer.transform([comment]).toarray()
        
        # 预测结果
        prediction = model.predict(comment_vector)
        
        return prediction[0]
    except Exception as e:
        return f"分析错误: {str(e)}"

def main():
    # 侧边栏
    with st.sidebar:
        st.title("YouTube评论垃圾检测系统")
        st.image("https://img.icons8.com/color/96/000000/youtube-play.png", width=100)
        
        # 模型选择
        model_version = st.selectbox(
            "选择模型版本",
            [1, 2, 3, 4, 5],
            index=0,
            format_func=lambda x: f"模型 v{x} ({'伯努利朴素贝叶斯' if x == 1 else '多项式朴素贝叶斯'})"
        )
        
        st.markdown("---")
        st.markdown("### 关于")
        st.info(
            "这个应用使用机器学习模型来检测YouTube评论中的垃圾内容。"
            "它基于朴素贝叶斯算法训练，可以识别潜在的垃圾评论。"
        )
        
        st.markdown("---")
        st.markdown("### 统计数据")
        st.write("已检测评论数: " + str(st.session_state.get('total_tested', 0)))
        st.write("识别垃圾评论数: " + str(st.session_state.get('spam_count', 0)))
        st.write("正常评论数: " + str(st.session_state.get('not_spam_count', 0)))

    # 主页面
    st.title("YouTube评论垃圾检测系统")
    st.subheader("输入评论进行分析")

    # 初始化会话状态变量
    if 'total_tested' not in st.session_state:
        st.session_state.total_tested = 0
    if 'spam_count' not in st.session_state:
        st.session_state.spam_count = 0
    if 'not_spam_count' not in st.session_state:
        st.session_state.not_spam_count = 0
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # 加载模型
    model, vectorizer, status = load_model_and_vectorizer(model_version)
    
    if isinstance(status, str):
        st.error(f"模型加载错误: {status}")
    else:
        # 用户输入
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "请输入YouTube评论文本:",
                height=150,
                placeholder="在这里输入评论文本来分析是否为垃圾评论..."
            )
        
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            analyze_button = st.button("分析评论", use_container_width=True)
            clear_button = st.button("清除", use_container_width=True)
            
            if clear_button:
                user_input = ""
                st.rerun()
        
        # 分析评论
        if analyze_button and user_input:
            with st.spinner("正在分析评论..."):
                # 添加一点延迟以增强用户体验
                time.sleep(0.5)
                
                # 预测结果
                result = predict_comment(user_input, model, vectorizer)
                
                # 更新统计信息
                st.session_state.total_tested += 1
                
                # 显示结果
                if result == "SPAM COMMENT":
                    st.markdown("<div class='spam-result spam'>🚨 结果: 垃圾评论 🚨</div>", unsafe_allow_html=True)
                    st.session_state.spam_count += 1
                    result_text = "垃圾评论"
                elif result == "NOT A SPAM COMMENT":
                    st.markdown("<div class='spam-result not-spam'>✓ 结果: 正常评论</div>", unsafe_allow_html=True)
                    st.session_state.not_spam_count += 1
                    result_text = "正常评论"
                else:
                    st.warning(f"结果: {result}")
                    result_text = result
                
                # 添加到历史记录
                st.session_state.history.append({
                    "评论": user_input,
                    "结果": result_text,
                    "时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # 标签页
        tab1, tab2, tab3 = st.tabs(["历史记录", "统计分析", "使用说明"])
        
        with tab1:
            if st.session_state.history:
                history_df = pd.DataFrame(st.session_state.history)
                st.dataframe(history_df, use_container_width=True)
                
                if st.button("清除历史记录"):
                    st.session_state.history = []
                    st.session_state.total_tested = 0
                    st.session_state.spam_count = 0
                    st.session_state.not_spam_count = 0
                    st.rerun()
            else:
                st.info("暂无历史记录。分析评论后将显示在这里。")
        
        with tab2:
            if st.session_state.total_tested > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # 饼图
                    fig, ax = plt.subplots(figsize=(5, 5))
                    labels = ['垃圾评论', '正常评论']
                    sizes = [st.session_state.spam_count, st.session_state.not_spam_count]
                    colors = ['#ff9999', '#66b3ff']
                    explode = (0.1, 0)
                    
                    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                            shadow=True, startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                
                with col2:
                    # 统计信息
                    st.markdown("### 详细统计")
                    st.markdown(f"**总检测评论数:** {st.session_state.total_tested}")
                    st.markdown(f"**垃圾评论数:** {st.session_state.spam_count} ({st.session_state.spam_count/st.session_state.total_tested*100:.1f}%)")
                    st.markdown(f"**正常评论数:** {st.session_state.not_spam_count} ({st.session_state.not_spam_count/st.session_state.total_tested*100:.1f}%)")
            else:
                st.info("暂无统计数据。分析评论后将显示统计信息。")
        
        with tab3:
            st.markdown("""
            ### 使用说明
            
            1. **选择模型版本**: 在左侧边栏中选择要使用的模型版本。
            2. **输入评论**: 在文本框中输入要分析的YouTube评论。
            3. **分析评论**: 点击"分析评论"按钮来检测评论是否为垃圾内容。
            4. **查看结果**: 结果将显示在文本框下方。
            5. **历史记录**: 在"历史记录"标签页中查看之前分析过的评论。
            6. **统计分析**: 在"统计分析"标签页中查看垃圾评论的统计信息。
            
            ### 关于模型
            
            - **模型v1**: 使用伯努利朴素贝叶斯算法，适合二进制特征。
            - **模型v2-v5**: 使用多项式朴素贝叶斯算法，适合词频特征。
            
            ### 注意事项
            
            - 该系统是一个辅助工具，可能不会100%准确。
            - 结果仅供参考，最终判断应结合具体情况。
            """)

if __name__ == "__main__":
    main() 