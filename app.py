# %%writefile app.py%
import streamlit as st
import pickle
import openpyxl
import xlrd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# loading the trained model
model = pickle.load(open('Sportpickle.pkl','rb'))


def main():
    html_temp = """ 
    <div style ="background-color:#002E6D;padding:20px;font-weight:15px"> 
    <h1 style ="color:white;text-align:center;"> Okoye-Nobert Sport Prediction Model</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    default_value_goes_here = ""

    uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")

    global dataframe
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        dataframe = df

    result = ""

    if st.button("Predict"):
#       arr = dataframe.columns

#       for i in arr:
#           notnull = dataframe[i][dataframe[i].notnull()]
#           min = notnull.min()
#           dataframe[i].replace(np.nan, min, inplace=True)

#       scaler = StandardScaler()
#       scaler.fit(dataframe)
#       featureshost = scaler.transform(dataframe)
      prediction = model.predict(dataframe)

      result = prediction
      st.write(result)


if __name__ == '__main__':
    main()
