# AI-Driven Scientific Hypothesis Generator and Statistical Validation (CLI Version)

This project generates research hypotheses from scientific abstracts using NLP models like BART and evaluates them with cosine similarity.

## Files
- `hypothesis_generator.py`: Main script for hypothesis generation and evaluation.
- `trained_model.pkl`: Pretrained classification model.
- `scientific_data_5000.csv`: Dataset of abstracts and categories.

## The model:
- Predicts the category of each scientific abstract
- Generates a plausible research hypothesis
- Computes cosine similarity between hypothesis and input
- Evaluates statistical confidence using T-Test and Z-Test

## Tech Stack
- Python 3.x
- scikit-learn (classification)
- HuggingFace Transformers (hypothesis generation)
- Sentence Transformers (similarity analysis)
- SciPy & NumPy (statistical testing)
- seaborn & matplotlib (visualizations)

## Output
The model predicts the category of a scientific abstract, generates a plausible hypothesis, computes similarity, and evaluates statistical confidence.

## 🛠Installation
Install the required packages using pip:

```bash
pip install transformers sentence-transformers pandas seaborn scikit-learn matplotlib joblib

