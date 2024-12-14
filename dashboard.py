import openai

openai.api_key = "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"  # Replace with your API key

models = openai.Model.list()
for model in models["data"]:
    print(model["id"])
