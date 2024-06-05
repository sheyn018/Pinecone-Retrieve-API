from transformers import AutoModel, AutoTokenizer

# Define the model name
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save the model and tokenizer to a directory
model.save_pretrained("all-MiniLM-L6-v2")
tokenizer.save_pretrained("all-MiniLM-L6-v2")
