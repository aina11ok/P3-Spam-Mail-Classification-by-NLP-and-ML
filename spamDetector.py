import streamlit as st
import pickle

# Load the model and vectorizer
try:
    model = pickle.load(open('spam123.pkl', 'rb'))
    cv = pickle.load(open('vec123.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    raise e

def main():
    st.title("Email Spam Classification Application")
    st.write("This is a machine learning application to classify emails into spam and genuine.")
    
    st.subheader("Classification")
    
    # Text area for user input
    user_input = st.text_area("Enter an email to classify", height=300)

    # Button to trigger classification
    if st.button("Classify"):
        if user_input:
            # Transform the user input into the format the model expects
            data = [user_input]
            vec = cv.transform(data).toarray()

            # Predict whether it's spam or not
            result = model.predict(vec)

            # Show result
            if result[0] == 0:
                st.success("This is Not A Spam Email")
            else:
                st.error("This is a Spam Email")
        else:
            st.warning("Please enter an email to classify.")

if __name__ == "__main__":
    main()
