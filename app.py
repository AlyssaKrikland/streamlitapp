import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import shap

df = pd.read_csv("traindata.csv")
df["hepatic_impairment"]= df["abbasic"]
df["trauma"] = df["accident"]

title = "Online calculator for predicting tigecycline-associated drug-induced liver injury"

model_path = "lgbm_model.pkl"
model = joblib.load(model_path)
names = model.feature_names_in_

BOOL = {"Yes":1, "No":0}

st.set_page_config(page_title=f"{title}", layout="wide", page_icon="🖥️")

explainer = shap.TreeExplainer(model, data=df[names], model_output='probability')

st.markdown("""
<style>
.index_name, [scope="row"] {
    display: none;
}
th {
    background: #3478CE;
}
th p {
    color: white;
}
th p, tr p{
    text-align: center;
    font-weight: bold;
}
[kind="primaryFormSubmit"] {
    background: #3478CE;
    border-color: #3478CE;
}
[kind="primaryFormSubmit"]:hover {
    background: #3478CE;
    border-color: #3478CA;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f'''
    <h1 style="text-align: center; font-size: 26px; font-weight: bold; color: white; background: #3478CE; border-radius: 0.5rem; margin-bottom: 15px;">
        {title}
    </h1>''', unsafe_allow_html=True)

data = {}
with st.form("inputform"):
    st.markdown("""<div style="color: red; font-size: 18px; text-align: center; border-bottom: 1px solid black; margin-bottom: 10px;">Hovering the mouse over the question mark displays detailed help for the corresponding input.</div>""", unsafe_allow_html=True)
    
    col = st.columns(3)
    
    data["drink"] = BOOL[col[0].selectbox("**History of alcohol use - :red[drink]**", ["Yes", "No"], index=0)]
    data["trauma"] = BOOL[col[1].selectbox("**Trauma as primary admission diagnosis - :red[trauma]**", ["Yes", "No"], index=1)]
    data["drugs"] = col[2].number_input("**Number of concomitant hepatotoxic drugs (count) - :red[drugs]**", value=2, step=1, min_value=0, help="Enter as an integer value (e.g., 3)")
    data["duration"] = col[0].number_input("**Duration of tigecycline therapy (days) - :red[duration]**", value=7, min_value=3, step=1, help="Enter as an integer value ≥3 (e.g., 14)")
    data["hemodialysis"] = BOOL[col[1].selectbox("**Receiving hemodialysis - :red[hemodialysis]**", ["Yes", "No"], index=0)]
    data["hepatic_impairment"] = BOOL[col[2].selectbox("**Abnormal baseline liver function - :red[hepatic_impairment]**", ["Yes", "No"], help="Any of ALT, ALP, or total bilirubin above the upper limit of normal at baseline.", index=0)]
    data["INR"] = col[0].number_input("**International normalized ratio - :red[INR]**", value=1.16, step=0.01, min_value=0.00, help="Enter as a decimal with two decimal places (e.g., 1.12)")
    

    c1 = st.columns(3)
    bt = c1[1].form_submit_button("**Start prediction**", use_container_width=True, type="primary")
    
if "predata" not in st.session_state:
    st.session_state.predata = data
else:
    pass


restext = [
    "This patient is classified as being at high risk of tigecycline-associated drug-induced liver injury.Closer monitoring of liver function tests should be considered.",
    "This patient is not classified as being at high risk of tigecycline-associated drug-induced liver injury. Liver function can be monitored as clinically indicated."
]

def prefun():
    pred_data = pd.DataFrame([st.session_state.predata])
    pred_data = pred_data[names]
    
    pred_data1 = pred_data.copy()
    
    for i in pred_data1.columns.tolist():
        pred_data1[i] = pred_data1[i].apply(lambda x: str(round(x, 2)) if isinstance(x, float) else str(x))
    
    with st.expander("**Current entered values**", True):
        st.table(pred_data1)

    res = model.predict(pred_data, output_margin=True)
    r_p = float(model.predict_proba(pred_data)[0][1])
    proba = round(r_p*100, 2)
    
    with st.expander("**Predict result**", True):
        st.info(f"**Model optimal threshold 0.154**")
        st.markdown(f'''
             <div style="text-align: center; font-size: 22px; color: black; margin-bottom: 5px; font-family: Times New Roman; border-bottom: 1px solid black;">
             Based on the entered values, the predicted probability of tigecycline-associated drug-induced liver injury in this patient is {proba}%
             </div>''', unsafe_allow_html=True)
        
        shap_values = explainer.shap_values(pred_data)
        plt.figure()
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            pred_data1.iloc[0],
            matplotlib=True,
            show=False
        )
        
        for t in plt.gca().texts:
            if "=" in t.get_text():
                t.set_rotation(0)
        
        col = st.columns([1, 6, 1])
        col[1].pyplot(plt.gcf(), use_container_width=True)
        
        if r_p>0.154:
            st.markdown(f"""<div style="color: black; font-size: 18px; text-align: center; font-weight: bold;">{restext[0]}</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="color: black; font-size: 18px; text-align: center; font-weight: bold;">{restext[1]}</div>""", unsafe_allow_html=True)
        
        st.markdown("""<div style="color: red; font-size: 18px; text-align: center;">Note: These results are for auxiliary reference only and must not be used as a standalone basis for diagnosis.</div>""", unsafe_allow_html=True)
        
if bt:
    st.session_state.predata = data
    prefun()
else:
    prefun()