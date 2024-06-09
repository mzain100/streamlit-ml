import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#3872fb;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Census Income Prediction App </h1>
		    <h4 style="color:white;text-align:center;">Team 4 </h4>
		    </div>
            """

desc_temp = """
            ### Census Income
            The goal is to create a predictive model that accurately determines if an individual earns more than $50K annually. This can be used for various business applications, such as targeted marketing, financial services, and social studies.
            #### Data Source
            - https://www.kaggle.com/code/tawfikelmetwally/census-income-analysis-and-modeling/input
            #### App Content
            - Machine Learning Section
            #### Model: Random Forest
            """

def main():

    stc.html(html_temp)
    
    menu = ['Home', 'Machine Learning']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.markdown(desc_temp)
    elif choice == "Machine Learning":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()


if __name__ == '__main__':
    main()
