import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="deep")

# 初始化 session state
if 'data_condition' not in st.session_state:
    st.session_state.data_condition = False
if 'df' not in st.session_state:
    st.session_state.df = None

use_default_data = st.button("使用默认数据")

uploaded_file = st.file_uploader("上传 CSV 或 XLSX 文件", type=["csv", "xlsx"])

if use_default_data:
    st.session_state.df = pd.read_csv("data_alive.csv")
    st.write('数据读取成功！')
    st.write(st.session_state.df.head(5))
    st.session_state.data_condition = True
elif uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write('数据读取成功！')
        st.session_state.data_condition = True
    else:
        st.session_state.df = pd.read_excel(uploaded_file)
        st.write('数据读取成功！')
        st.session_state.data_condition = True

if st.session_state.data_condition:
    df = st.session_state.df
    id_var = st.selectbox("选择ID变量", df.columns)
    target_var = st.selectbox("选择因变量", [col for col in df.columns if col != id_var])
    default_features = [col for col in df.columns if col not in [id_var, target_var]]
    selected_features = st.multiselect("选择自变量", default_features, default=default_features)

    fit_begin = st.button("开始拟合模型！")

    if fit_begin:
        df_s = df[selected_features + [id_var, target_var]]
        dataset = TabularDataset(df_s)
        predictor = TabularPredictor(label=target_var, problem_type='regression').fit(dataset, hyperparameters={'GBM': {}, 'XGB': {}})
        predictions = predictor.predict(dataset)
        st.write('模型拟合成功!')

        def plot_feature_importance(model_name):
            importance_df = predictor.feature_importance(data=dataset, model=model_name)
            top10_features = importance_df.head(10)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top10_features['importance'], y=top10_features.index)
            plt.title(f'{model_name} Top 10 Features')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            st.pyplot(plt.gcf())

        plot_feature_importance('WeightedEnsemble_L2')
else:
    st.write("请上传文件或者使用默认数据")
