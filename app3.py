import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="deep")

# 初始化 session state
if 'data_condition' not in st.session_state:
    st.session_state.data_condition = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_fitted' not in st.session_state:
    st.session_state.model_fitted = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = False
if 'model_condition' not in st.session_state:
    st.session_state.model_condition = False
if 'id_var' not in st.session_state:
    st.session_state.id_var = False
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = False

# 创建一个导航菜单
page = st.sidebar.selectbox("导航", ["Home", "模型拟合", "模型解释","单条数据解释"])

if page == "Home":
    st.title("欢迎使用数据分析工具")
    st.write("""
    ### 介绍
    这是一个用于数据分析和机器学习模型拟合与解释的工具。你可以通过侧边栏导航到不同的功能页面。

    ### 使用说明
    1. 在模型拟合页面，你可以上传自己的数据集，选择自变量和因变量，并进行模型拟合。
    2. 在模型解释页面，你可以查看模型的解释结果。
    3. 在单条数据解释页面，你可以查看模型对于单条数据给出判断结果的依据。
    
    请在侧边栏选择相应的功能页面开始使用。
    """)

elif page == "模型拟合":
    st.title("模型拟合")

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
        st.session_state.id_var=id_var
        target_var = st.selectbox("选择因变量", [col for col in df.columns if col != id_var])
        default_features = [col for col in df.columns if col not in [id_var, target_var]]
        selected_features = st.multiselect("选择自变量", default_features, default=default_features)
        st.session_state.selected_features=selected_features
        fit_begin = st.button("开始拟合模型！")

        if fit_begin:
            df_s = df[selected_features + [target_var]]
            st.session_state.dataset = TabularDataset(df_s)
            predictor = TabularPredictor(label=target_var, problem_type='regression').fit(st.session_state.dataset, hyperparameters={'GBM': {}, 'XGB': {}})
            st.session_state.predictor = predictor
            st.session_state.model_fitted = True
            st.write('模型拟合成功!')

            def plot_feature_importance(model_name):
                importance_df = predictor.feature_importance(data=st.session_state.dataset, model=model_name)
                top10_features = importance_df.head(10)
                plt.figure(figsize=(10, 6))
                sns.barplot(x=top10_features['importance'], y=top10_features.index)
                plt.title(f'{model_name} Top 10 Features')
                plt.xlabel('Importance')
                plt.ylabel('Features')
                st.pyplot(plt.gcf())
            
            # 在一个容器中显示特征重要性图
            with st.container():
                plot_feature_importance('WeightedEnsemble_L2')
            
            st.session_state.model_condition = True
    else:
        st.write("请上传文件或者使用默认数据")

elif page == "模型解释":
    st.title("模型解释")

    if st.session_state.model_condition:
        random_state1 = st.number_input("请输入背景数据随机种子 (random_state)", value=42)
        background_sample_size = st.slider("请选择背景数据采样数", min_value=100, max_value=15000, value=7500, step=100)
        random_state2 = st.number_input("请输入样本数据随机种子 (random_state)", value=42)
        sample_data_size = st.slider("请选择样本数据采样数", min_value=1, max_value=50, value=25, step=1)
        shap_explain = st.button("开始解释！")
        if shap_explain:
            model_to_explain = st.session_state.predictor._trainer.load_model('WeightedEnsemble_L2')
            background_data = st.session_state.dataset.sample(n=background_sample_size, random_state=random_state1)
            explainer = shap.Explainer(model_to_explain.predict, background_data)
            sample_data = st.session_state.dataset.sample(n=sample_data_size,random_state=random_state1)
            shap_values = explainer(sample_data)
            
            # 在一个新容器中显示 SHAP 图
            with st.container():
                st.subheader('summary_plot')
                shap.summary_plot(shap_values, sample_data)
                st.pyplot(bbox_inches='tight')
                st.subheader('bar_plot')
                shap.plots.bar(shap_values)
                st.pyplot(bbox_inches='tight')
    else:
        st.write("请先完成模型拟合")
elif page == "单条数据解释":
    st.title("单条数据解释")

    if st.session_state.model_condition:
        df = st.session_state.df
        
        if st.session_state.id_var:
            selected_id = st.text_input("输入数据ID")
            
            if selected_id:
                try:
                    selected_id = type(df[st.session_state.id_var].iloc[0])(selected_id)  # 转换输入的ID类型以匹配数据类型
                    single_data = df[df[st.session_state.id_var] == selected_id]
                    if not single_data.empty:
                        random_state = st.number_input("请输入背景数据随机种子 (random_state)", value=42)
                        background_sample_size = st.slider("请选择背景数据采样数", min_value=5000, max_value=20000, value=10000, step=1000)
                        shap_explain_single = st.button("解释该数据")
                        st.session_state.selected_features
                        if shap_explain_single:
                            model_to_explain = st.session_state.predictor._trainer.load_model('WeightedEnsemble_L2')
                            background_data = st.session_state.dataset.sample(n=background_sample_size, random_state=random_state)
                            explainer = shap.Explainer(model_to_explain.predict, background_data)
                            shap_values_single = explainer(single_data[st.session_state.selected_features])
                            
                            # 在一个新容器中显示 SHAP force_plot
                            with st.container():
                                st.subheader('force_plot')
                                shap.force_plot(explainer.expected_value, shap_values_single.values, single_data[selected_features], matplotlib=True)
                                st.pyplot(bbox_inches='tight')
                    else:
                        st.write("未找到相应的数据，请检查ID是否正确。")
                except ValueError:
                    st.write("输入的ID格式不正确，请输入有效的ID。")
    else:
        st.write("请先完成模型拟合")
    

