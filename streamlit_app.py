import numpy as np
import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings("ignore")
model = pickle.load(open("Ad_classifier.pkl", "rb"))


st.set_page_config(page_title="Social Application") #tab title

#prediction function
def prediction(age,salary):
    input_data = np.asarray([age,salary])
    input_data = input_data.reshape(1,-1)
    prediction = model.predict(input_data)
    return prediction[0]

def main():

    # titling your page
    st.title("cancer Prediction App")
    st.write("Demo application ")

    #getting the input
    age = st.text_input("Enter age")
    salary = st.text_input("Enter salary")
    

    #predict value
    diagnosis = ""

    if st.button("Predict"):
        diagnosis = prediction(age,salary)
        if diagnosis=="1":
            st.info("This person will buy the product.")
        elif diagnosis=="0":
                st.info("This person willnot buy the product.")      
    else:
        st.error("Try again!")

            
        
        
        st.write("Project by Abhilasha Waje")
        


if __name__=="__main__":
    main()
