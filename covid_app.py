import streamlit as st
from sklearn import tree
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd


# Adding Title
st.title("COVID 19 SYMPTOM PREDICTION")

# DATA
data=pd.read_csv(r'C:\Users\HP\Desktop\Data\new_covid.csv')

x=data[["cough","fever","sore_throat","shortness_of_breath","head_ache","age_60_and_above","gender","test_indication"]]
y=data[["corona_result"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1, stratify = y)

# fitting the model
model = tree.DecisionTreeClassifier(max_depth=6,min_samples_leaf=2,max_leaf_nodes=15)

model.fit(x_train, y_train)

# making Prediction

y_pred = model.predict(x_test)


gender_type = st.sidebar.selectbox("Select gender", ("male", "female"))
def get_gender(gender):
    if gender == "male":
        return 1
    else:
        return 0

cough_type = st.sidebar.selectbox("Cough", ("yes", "no"))
def get_cough(cough):
    if cough == "yes":
        return 1
    else:
        return 0

fever_type = st.sidebar.selectbox("fever", ("yes", "no"))

def get_fever(fever):
    if fever == "yes":
        return 1
    else:
        return 0
sore_throat_type = st.sidebar.selectbox("sore_throat",("yes","no"))
def get_sore_throat(sore_throat):
    if sore_throat == "yes":
        return 1
    else:
        return 0

shortness_of_breath_type = st.sidebar.selectbox("shortness_of_breath",("yes","no"))
def get_shortness_of_breath(shortness_of_breath):
    if shortness_of_breath == "yes":
        return 1
    else:
        return 0

head_ache_type = st.sidebar.selectbox("head_ache",("yes","no"))
def get_head_ache(head_ache):
    if head_ache == "yes":
        return 1
    else:
        return 0

age_60_and_above_type = st.sidebar.selectbox("age_60_and_above",("yes","no"))
def get_age_60_and_above(age_60_and_above):
    if age_60_and_above == "yes":
        return 1
    else:
        return 0

test_indication_type = st.sidebar.selectbox("test_indication",("yes","no"))
def get_test_indication(test_indication):
    if test_indication == "yes":
        return 1
    else:
        return 0

import numpy as np
# Prediction
new_pr= np.array([[get_cough(cough_type),get_fever(fever_type),get_sore_throat(sore_throat_type),
                   get_shortness_of_breath(shortness_of_breath_type),get_head_ache(head_ache_type),
                   get_age_60_and_above(age_60_and_above_type),get_gender(gender_type),
                   get_test_indication(test_indication_type)]])
new_pr = new_pr.reshape(1, -1)


if st.button("Predict"):
    price = model.predict(new_pr)
    #st.write("diabetes:", price)
    if price==1:
        st.write("""you have the covid symptoms.
         the following suggestions are for you:""")
    else:
        st.write("""you does not have any covid symptoms.
        you are suggested the following tips to be safe and healthy""")

