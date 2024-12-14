import openai
import streamlit as st

# Load the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to test OpenAI API
def test_openai_api():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use gpt-3.5-turbo for this test
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain the significance of AI in modern technology."},
            ],
        )
        # Return the assistant's response
        return response["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        return f"OpenAI API error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Streamlit interface
st.title("OpenAI API Test")
result = test_openai_api()
st.write("API Response:")
st.write(result)
