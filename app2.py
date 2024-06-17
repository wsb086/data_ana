import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

# 上传CSV文件
uploaded_file = st.file_uploader("上传CSV文件", type="csv")

if uploaded_file is not None:
    # 读取CSV数据
    data = pd.read_csv(uploaded_file)
    
    # 假设第一列是类别，后面的列是特征
    labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]
    
    # 进行t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    # 将t-SNE结果和标签装入一个DataFrame
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['Label'] = labels
    
    # 可视化t-SNE结果
    fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2', color='Label', hover_data=[labels.name])
    st.plotly_chart(fig)

    # 显示详细信息
    selected_points = st.selectbox("选择一个点查看详细信息", tsne_df.index)
    if selected_points is not None:
        st.write("详细信息:")
        st.write(data.iloc[selected_points])
