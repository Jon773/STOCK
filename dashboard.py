import openai
import streamlit as st

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Ensure your key is in Streamlit Secrets

# Define a simple prompt
prompt = "Explain the significance of AI in modern technology in a brief paragraph."

# Call OpenAI's GPT API
def test_openai_api():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if GPT-4 is unavailable
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        # Return the response text
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit interface to display the result
st.title("OpenAI API Test")
result = test_openai_api()
st.write("API Response:")
st.write(result)
