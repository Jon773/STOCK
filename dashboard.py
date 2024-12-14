import streamlit as st

# Debug Streamlit secrets
st.title("Secrets Debugging")

# Check if secrets are loaded
try:
    st.write("Secrets Loaded:")
    st.write(st.secrets)  # Display all secrets
except Exception as e:
    st.error(f"Error: {e}")
