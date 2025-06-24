# interpretability.py

import shap
import lime
import lime.lime_text
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# -------------------------
# Load your trained model and tokenizer
# -------------------------
MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'  # Replace with your model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set model to evaluation mode
model.eval()

# -------------------------
# Define your class labels
# -------------------------
unique_labels = ['Negative', 'Positive']  # Adjust according to your model's classes

# -------------------------
# Define your prediction function
# -------------------------
def model_predict(texts):
    """
    Takes a list of texts and returns model probabilities.
    """
    # Ensure input is a list of strings
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().cpu().numpy()

# -------------------------
# Example texts for explanation
# -------------------------
texts = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had."
]

# -------------------------
# SHAP explanation function
# -------------------------
def explain_shap(texts):
    print("Starting SHAP explanation...")
    # Use a valid, neutral background sentence
    background = ['This is a neutral statement suitable for background.']
    # Create SHAP explainer
    explainer = shap.Explainer(model_predict, tokenizer)
    # Generate SHAP values for the background
    shap_values = explainer(background)
    # Plot summary for the background example
    shap.summary_plot(shap_values, features=background)
    print("SHAP explanation completed.")

# -------------------------
# LIME explanation function
# -------------------------
def explain_lime(text):
    print("Starting LIME explanation...")
    explainer = lime.lime_text.LimeTextExplainer(class_names=unique_labels)
    exp = explainer.explain_instance(text, model_predict, num_features=10)
    exp.show_in_notebook(show_table=True)
    print("LIME explanation completed.")

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    print("Script started.")
    # Generate SHAP explanations
    explain_shap(texts)
    # Generate LIME explanation for the first text
    explain_lime(texts[0])
    print("Script finished.")