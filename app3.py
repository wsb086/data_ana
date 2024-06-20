import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
import shap
import matplotlib.pyplot as plt

uploaded_file = st.file_uploader("上传 CSV 或 XLSX 文件", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    df = pd.read_excel("default.xlsx")  # 使用默认文件

# 选择ID变量
id_var = st.selectbox("选择ID变量", df.columns)

# 选择因变量
target_var = st.selectbox("选择因变量", [col for col in df.columns if col != id_var])

# 选择自变量
default_features = [col for col in df.columns if col not in [id_var, target_var]]
selected_features = st.multiselect("选择自变量", df.columns, default=default_features)
