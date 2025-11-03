from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Charger un modèle et son tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Utiliser le modèle
text = "I love using Hugging Face models!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
