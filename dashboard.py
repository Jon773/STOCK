import openai

openai.api_key = "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"  # Replace with your API key

try:
    # List available models
    models = openai.Model.list()
    print("Available models:")
    for model in models["data"]:
        print(model["id"])
except Exception as e:
    print(f"Error: {e}")
