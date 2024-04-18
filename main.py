import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

def get_clean_data():
  data = pd.read_csv('./data/data_transform.csv').drop(['Exited'],axis=1)
  return data


def add_sidebar():
    st.sidebar.header("Churn Predictors")
    data = get_clean_data()
    
    slider_labels_float = list(data.select_dtypes(include=['float']).columns)
    slider_labels_float = [(item, item) for item in slider_labels_float]
    slider_labels_int = list(data.select_dtypes(include=['int']).columns)
    slider_labels_int = [(item, item) for item in slider_labels_int]
    slider_labels_str = list(data.select_dtypes(include=['object', 'category']).columns)
    slider_labels_str = [(item, item) for item in slider_labels_str]
    
    input_dict = {}

    for label, key in slider_labels_float:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value= float(0),
            max_value=float(data[label].max()),
            value=float(data[label].mean()))
        
    for label, key in slider_labels_int:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value= 0,
            max_value= int(data[label].max()),
            value= int(data[label].median()))
        
    for label, key in slider_labels_str:
        input_dict[key] = st.sidebar.radio(
        label, 
        list(data[label].unique()),
        horizontal = True)
    #Order based on data columns
    input_dict = {col: input_dict[col] for col in data.columns}   
           
    return input_dict

def add_predictions(input_data):
    model = pickle.load(open("./model/model.pkl", "rb"))
    input_array = pd.DataFrame(input_data, index=[0]) 
    prediction = model.predict(input_array)
    st.write("The customer is a:")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis churner'>Non Churner</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis nonchurner'>Churner</span>", unsafe_allow_html=True)
        
    st.write("Probability of being a churner:", round(model.predict_proba(input_array)[0][0]*100,3))
    st.write("Probability of being a non-churner: ", round(model.predict_proba(input_array)[0][1]*100,3))


def get_scaled_values(input_data):
  data = get_clean_data()
  numeric_cols = list(data.select_dtypes(include=['float','int']).columns)
  input_dict_numeric = {key: value for key, value in input_data.items() if key in numeric_cols}
  scaled_dict = {}
  
  for key, value in input_dict_numeric.items():
    max_val = data[key].max()
    min_val = data[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  return scaled_dict

def get_radar_chart(input_data):
  data = get_clean_data()
  input_data = get_scaled_values(input_data)
  numeric_cols = list(data.select_dtypes(include=['float','int']).columns)

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[input_data[col] for col in numeric_cols],
        theta=numeric_cols,
        fill='toself',
        name='Value',
        line=dict(color='#ee4035')
  ))
  fig.update_layout(
    polar=dict(
    bgcolor='rgba(0,0,0,0)',  # Set background color of the polar plot area to transparent
        ),
    paper_bgcolor='rgba(0,0,0,0)',
    width = 300,
    height = 300,
    margin=dict(l=0, r=0, t=0, b=0)
    )

  return fig

def profile_report():
    data = get_clean_data()
    profile = ProfileReport(data, title="Pandas Profiling Report")
    return profile


def main():
    selected = option_menu(
       menu_title = "",
       options = ["Home","EDA","Links",],
       icons = ["house",'book','envelope'],
       menu_icon = 'cast',
       default_index = 0,
       orientation = "horizontal"
   )    
    with open("./assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)    
##-------------------------------------Homepage-------------------------------------
    if selected == "Home":     
        input_data = add_sidebar() 
        with st.container():
            st.markdown("<h1 style='text-align: center; font-size: 30px;'>Churn Prediction</h1>", unsafe_allow_html=True)
            st.write("Using machine learning, the app analyzes historical data to predict potential churners. With interactive sidebar inputs, explore scenarios to optimize retention strategies. While a sample, it highlights the value of proactive customer management and predictive analytics for business success.")
        
        col1, col2 = st.columns([4,2])
        with col1:
            st.write("Some characteristics of the customer")
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart,use_container_width=True,align='center')
        with col2:
            add_predictions(input_data)

##-------------------------------------EDA-------------------------------------    
    if selected == "EDA":
        profile = profile_report()
        with st.spinner("Generating Report....\nplease wait...."):
            st.components.v1.html(profile.to_html(), width=1000, height=1200, scrolling=True)  
 
##-------------------------------------Useful links-------------------------------------
    if selected == "Links":
        st.write("Please visit the link below to see the codebook for creating the model and the app you are currently viewing. The link showcases techniques such as feature selection, class imbalance handling, model selection, and hyperparameter tuning. All processes are integrated into pipelines to avoid data leakage. The SHAP explanation includes feature importance with a beeswarm plot, SHAP dependence plot with and without interaction to investigate relationships between variables, and interpretation of local prediction with SHAP waterfall plot and force plot.")        
        link = '[GitHub notebook](https://github.com/tuyennt812)'
        st.markdown(link, unsafe_allow_html=True)
        st.write("Please also visit the link below to see the Tableau visualization of the data")
if __name__ == '__main__':
  main()