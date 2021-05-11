
# -*- coding: utf-8 -*-
"""

@author: Pritesh Kumar
"""

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('logisticmodel.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('Classification Dataset3.csv')
X = dataset.iloc[:, 0:9].values
# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 1:8]) 
#Replacing missing data with the calculated mean value  
X[:, 1:8]= imputer.transform(X[:, 1:8])





# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()
dataset['label']=number.fit_transform(dataset['label'].astype('str'))



X = dataset.iloc[:, 0 : 9]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


def predict_note_authentication(meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange):
  output= model.predict(sc.transform([[meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange]]))
  print("The ", output)
  if output==[0]:
    prediction="person is Male"
  else:
    prediction="person is Female"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:#fa8072" >
   <div class="clearfix">           
   <div class="col-lg-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Pritesh Kumar (mid term 1) </p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering PIET,Jaipur</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Lab </p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Logistic Classification")
    meanfreq = st.number_input("Enter The meanfreq value",0,1)
    sd = st.number_input("Enter The sd value",0,1)
    median = st.number_input("Enter The median value",0,1)
    iqr = st.number_input("Enter The iqr value",0,1)
    skew = st.number_input("Enter The skew value",0,15)
    kurt = st.number_input('Enter the kurt value ',0,600)
    mode = st.number_input('Enter the mode value',0,1)
    centroid = st.number_input("Enter the centroid value",0,1)
    dfrange = st.number_input("Enter the dfrange value",0,5)
    if st.button("Predict"):
      result=predict_note_authentication(meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange)
      st.success('{} '.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Pritesh Kumar")
      st.subheader("Student , Poornima Institute Of Engineering And Technology")

if __name__=='__main__':
  main()
