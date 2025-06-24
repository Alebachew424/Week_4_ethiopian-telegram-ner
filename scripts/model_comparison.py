# Save as model_comparison.py

import matplotlib.pyplot as plt

# Assume you have eval results stored in a dictionary
results = {
    'XLM-Roberta': {'eval_f1': 0.85, 'eval_loss': 0.3},
    'DistilBERT': {'eval_f1': 0.82, 'eval_loss': 0.35},
    'mBERT': {'eval_f1': 0.83, 'eval_loss': 0.33}
}

models = list(results.keys())
f1_scores = [results[m]['eval_f1'] for m in models]
losses = [results[m]['eval_loss'] for m in models]

# Plot F1 scores
plt.figure(figsize=(8,4))
plt.bar(models, f1_scores, color='skyblue')
plt.ylabel('F1 Score')
plt.title('Model Performance Comparison')
plt.show()

# Plot Loss
plt.figure(figsize=(8,4))
plt.bar(models, losses, color='salmon')
plt.ylabel('Evaluation Loss')
plt.title('Model Loss Comparison')
plt.show()