import streamlit as st
import pandas as pd
from deepchecks import Dataset
from deepchecks.suites import data_integrity

# Streamlit application
def main():
    st.title("Deepchecks Data Integrity Suite Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type="csv")
    
    if uploaded_file is not None:
        # Read the Excel file
        df = pd.read_csv(uploaded_file)
        
        st.write("Dataframe Preview:")
        st.write(df.head())
        
        # Create a Deepchecks Dataset
        dataset = Dataset(df, label=None)
        
        # Run Data Integrity Suite
        suite = data_integrity()
        try:
            suite_result = suite.run(dataset)
        except BrokenPipeError as e:
            print(f"An error occurred: {e}")
        # Display the results
        st.write("Data Integrity Suite Results:")
        html_result = suite_result.save_as_html()

        # Display the HTML in Streamlit
        st.components.v1.html(html_result, height=800, scrolling=True)
        

if __name__ == "__main__":
    main()
