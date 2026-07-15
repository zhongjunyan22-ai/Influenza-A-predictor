# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st  

# 导入 joblib 库，用于加载和保存机器学习模型
import joblib  

# 导入 NumPy 库，用于数值计算
import numpy as np  

# 导入 Pandas 库，用于数据处理和操作
import pandas as pd  

# 导入 SHAP 库，用于解释机器学习模型的预测
import shap  

# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt  

# 从 LIME 库中导入 LimeTabularExplainer，用于解释表格数据的机器学习模型
from lime.lime_tabular import LimeTabularExplainer  

# 加载训练好的随机森林模型（RF.pkl）
model = joblib.load('RF.pkl')  

# 从 X_test.csv 文件加载测试数据，以便用于 LIME 解释器
X_test = pd.read_csv('X_test.csv')  

# 定义特征名称，对应数据集中的列名
feature_names = [  
    "Gender",       # 性别  
    "Age",       # 年龄
    "WBC",        #  
    "PNEUT",  #  
    "PLYMPH",      # 
    "PEOS",       #  
    "PBASO",   # 
    "PMONO",   # 
    "NEUT",     #  
    "LYMPH",   #  
    "EOS",     # 
    "BASO",        #   
    "MONO",   #   
    "RBC",       # 
    "HB",   #  
    "HCT",   # 
    "MCV",     # 
    "MCH",   #  
    "MCHC",     #  
    "RDW_CV",        #  
    "PLT",        # 
    "PCT",       # 
    "MPV",   # 
    "PDW",   #  
    "hs_CRP",     #   
]  

# Streamlit 用户界面
st.title("Influenza A predictor")  # 设置网页标题

# 性别：分类选择框（0：女性，1：男性）
Gender = st.selectbox("Gender:", options=[0, 1], format_func=lambda x: "male" if x == 1 else "female")  

# 数值型变量输入框
Age = st.number_input("Age:", min_value=0.00, max_value=120.00, value=8.00, step=0.01)
WBC = st.number_input("WBC:", min_value=0.00, max_value=200.00, value=5.69, step=0.01) 
PNEUT = st.number_input("PNEUT:", min_value=0.00, max_value=100.00, value=68.20, step=0.01) 
PLYMPH = st.number_input("PLYMPH:", min_value=0.00, max_value=100.00, value=20.91, step=0.01) 
PEOS = st.number_input("PEOS:", min_value=0.00, max_value=100.00, value=3.90, step=0.01) 
PBASO = st.number_input("PBASO:", min_value=0.00, max_value=100.00, value=0.01, step=0.01) 
PMONO = st.number_input("PMONO:", min_value=0.00, max_value=100.00, value=8.60, step=0.01) 
NEUT = st.number_input("NEUT:", min_value=0.00, max_value=100.00, value=3.88, step=0.01) 
LYMPH = st.number_input("LYMPH:", min_value=0.00, max_value=100.00, value=1.19, step=0.01) 
EOS = st.number_input("EOS:", min_value=0.00, max_value=100.00, value=0.22, step=0.01) 
BASO = st.number_input("BASO:", min_value=0.00, max_value=100.00, value=0.01, step=0.01) 
MONO = st.number_input("MONO:", min_value=0.00, max_value=100.00, value=0.49, step=0.01)    
RBC = st.number_input("RBC:", min_value=0.00, max_value=20.00, value=4.30, step=0.01) 
HB = st.number_input("HB:", min_value=0.00, max_value=200.00, value=120.00, step=0.01) 
HCT = st.number_input("HCT:", min_value=0.00, max_value=70.00, value=40.00, step=0.01) 
MCV = st.number_input("MCV:", min_value=0.00, max_value=100.00, value=84.00, step=0.01) 
MCH = st.number_input("MCH:", min_value=0.00, max_value=70.00, value=27.00, step=0.01) 
MCHC = st.number_input("MCHC:", min_value=0.00, max_value=500.00, value=328.00, step=0.01) 
RDW_CV = st.number_input("RDW_CV:", min_value=0.00, max_value=30.00, value=13.00, step=0.01) 
PLT = st.number_input("PLT:", min_value=0.00, max_value=900.00, value=330.00, step=0.01) 
PCT = st.number_input("PCT:", min_value=0.00, max_value=10.00, value=0.30, step=0.01 )
MPV = st.number_input("MPV:", min_value=0.00, max_value=30.00, value=9.30, step=0.01) 
PDW = st.number_input("PDW:", min_value=0.00, max_value=100.00, value=16.40, step=0.01) 
hs_CRP = st.number_input("hs_CRP:", min_value=0.00, max_value=900.00, value=0.36, step=0.01) 

# 处理输入数据并进行预测
feature_values = [Gender, Age, WBC, PNEUT, PLYMPH, PEOS, PBASO, PMONO, NEUT, LYMPH, EOS, BASO, MONO, RBC, HB, HCT, MCV, MCH, MCHC, RDW_CV, PLT, PCT, MPV, PDW, hs_CRP]  # 将用户输入的特征值存入列表
features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
     # 预测类别（0：无心脏病，1：有心脏病）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

     # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: Influenza A infection, 0: Non-influenza A infection)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为 1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of influenza A infection. "
            f"The model predicts that your probability of having influenza A infection is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    # 如果预测类别为 0（低风险）
    else:
        advice = (
            f"According to our model, you have a low risk of influenza A infection. "
            f"The model predicts that your probability of not having influenza A infection is {probability:.1f}%. "
            "However, if you and your child are in an area with a flu epidemic, please take daily protective measures, such as wearing masks regularly, washing hands frequently, following cough etiquette, and maintaining indoor air circulation. Additionally, it is recommended that all children aged 6 months and above who have no vaccination contraindications receive the seasonal flu vaccine annually."
        )
    # 显示建议
    st.write(advice)

    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    # 创建 SHAP 解释器，基于树模型（如随机森林）
    explainer_shap = shap.TreeExplainer(model)
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    # 根据预测类别显示 SHAP 强制图
    # 期望值（基线值）
    # 解释类别 1（患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # 期望值（基线值）
    # 解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Non-influenza A infection', 'Influenza A infection'],  # Adjust class names to match your classification task
        mode='classification'
    )
    
    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)
