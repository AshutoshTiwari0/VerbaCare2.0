import streamlit as st
import pandas as pd
import joblib
#import numpy as np


st.title("VerbaCare")
st.write("VerbaCare: Advanced natural language processing that detects signs of distress in text. AI technology which works alongside human support systems to identify those who may need help, turning words into timely intervention.")
st.image("Verbacare_image.jpeg",width=500)

#loading the pipeline
pipe_predict = joblib.load(open("suicide_pipeline.pkl", 'rb'))


#method for prediction
def predict_emotions(docx):
    results=pipe_predict.predict([docx])[0]
    if results==0:
        return "Non Sucidal Thought"
    else:
        return "Sucidal Thought"



def main():
    with st.form(key='emotion form'):
        raw_text=st.text_area("Enter Text Here")
        submit_text=st.form_submit_button(label='Submit')

    if submit_text:
        col1,col2=st.columns(2)

        prediction=predict_emotions(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

        with col2:
            st.success("Prediction")
            st.write(prediction)

if __name__=='__main__':
    main()