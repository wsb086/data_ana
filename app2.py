import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

# 上传CSV文件
uploaded_file = st.file_uploader("上传CSV文件", type="csv")

if uploaded_file is not None:
    # 读取CSV数据
    data = pd.read_csv(uploaded_file)
    
    # 假设最后一列是类别，前面的列是特征
    labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]
    
    # 进行t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    # 将t-SNE结果和标签装入一个DataFrame
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['Label'] = labels
    
    # 打印调试信息
    st.write("Data Preview:")
    st.write(data.head())
    st.write("t-SNE Results Preview:")
    st.write(tsne_df.head())
    

    # 可视化t-SNE结果
    fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2', color='Label')
    st.plotly_chart(fig)


# 如果没有上传文件，使用示例数据集
else:
    st.write("请上传一个CSV文件以进行t-SNE和可视化。")
    st.write("你可以使用以下示例数据集进行测试：")
    st.write("Iris 数据集 (下载链接: [iris.csv](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data))")
