import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor
warnings.filterwarnings('ignore')
# 设置 Seaborn 的主题和样式
sns.set_theme(style="whitegrid", palette="deep")
use_default_data = st.button("使用默认数据")

uploaded_file = st.file_uploader("上传 CSV 或 XLSX 文件", type=["csv", "xlsx"])
data_condition=0
if use_default_data:
    # 使用默认文件
    df = pd.read_csv("data_alive.csv")
    st.write('数据读取成功！')
    st.write(df.head(5))
    data_condition=1
else:
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.write('数据读取成功！')
            data_condition=1
        else:
            df = pd.read_excel(uploaded_file)
            st.write('数据读取成功！')
            data_condition=1
    else:
        df = None  # 如果没有上传文件且没有选择使用默认数据，那么数据框为空

# 检查数据框是否为空
if data_condition:
    # 选择ID变量
    df_s=None
    id_var = st.selectbox("选择ID变量", df.columns)
    
    # 选择因变量
    target_var = st.selectbox("选择因变量", [col for col in df.columns if col != id_var])
    
    # 选择自变量
    default_features = [col for col in df.columns if col not in [id_var, target_var]]
    selected_features = st.multiselect("选择自变量", default_features, default=[])
    df_s=df[selected_features+[id_var,target_var]]
    if df_s and selected_features!=[] and id_var and target_var:
        fit_begin = st.button("开始拟合模型！")
        if fit_begin:
            dataset = TabularDataset(df_s)
            predictor = TabularPredictor(label=target_var, problem_type='regression').fit(dataset,hyperparameters={'GBM':{},'XGB':{}})
            predictions = predictor.predict(dataset)
            st.write('模型拟合成功!')
            def plot_feature_importance(model_name):
                importance_df = predictor.feature_importance(data=dataset, model=model_name)
                top10_features = importance_df.head(10)
                myfig=sns.barplot(x=top10_features['importance'], y=top10_features.index)
                myfig.set_title(f'{model_name} Top 10 Features')
                myfig.set_xlabel('Importance')
                myfig.set_ylabel('Features')
                plt.figure(figsize=(10, 6))
                st.pyplot(plt.gcf())
            plot_feature_importance('WeightedEnsemble_L2')
    
else:
    st.write("请上传文件或者使用默认数据")

    

