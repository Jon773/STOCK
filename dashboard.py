import openai
import streamlit as st

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Ensure your OpenAI API key is stored in Streamlit Secrets

# Define a simple test function for the OpenAI API
def test_openai_api():
    try:
        # Use the new Chat API structure
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if GPT-4 is unavailable
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain the significance of AI in modern technology in a brief paragraph."},
            ],
        )
        # Extract the content of the response
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        # Handle errors and return them as output
        return f"Error: {str(e)}"

# Streamlit app to display the response
st.title("OpenAI API Test")
result = test_openai_api()
st.write("API Response:")
st.write(result)
