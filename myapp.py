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
        suite_result = suite.run(dataset)
        
        # Display the results
        st.write("Data Integrity Suite Results:")
        suite_result.show()
        suite_result.save_as_html('suite_result.html')
        
        # Display the HTML result in an iframe
        st.markdown(
            f'<iframe src="suite_result.html" width="100%" height="600"></iframe>',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
