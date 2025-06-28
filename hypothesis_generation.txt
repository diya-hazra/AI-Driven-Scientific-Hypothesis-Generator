import joblib
import pandas as pd
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from scipy import stats

# Load dataset
dataset_path = r"C:\Users\mdfra\Documents\ML_Project\scientific_data_5000.csv"
df = pd.read_csv(dataset_path)

# Input and target columns
X = df["Abstract"]
y = df["Category"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load classification model
model = joblib.load("trained_model.pkl")

# Load BART hypothesis generation model
hypothesis_generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Load sentence transformer for similarity
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to generate hypothesis
def generate_hypothesis(abstract):
    result = hypothesis_generator(f"Generate a research hypothesis from: {abstract}", max_length=50, truncation=True)
    return result[0]["generated_text"]

# Function to calculate cosine similarity
def compute_similarity(hypothesis, abstract):
    emb1 = similarity_model.encode(hypothesis, convert_to_tensor=True)
    emb2 = similarity_model.encode(abstract, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

# Generate predictions and hypotheses
similarities = []

print("\n🔍 Generating hypotheses & computing similarity scores...\n")
for i in range(30):  # use 30 samples for statistical validity
    abstract = X_test.iloc[i]
    predicted_category = model.predict([abstract])[0]
    hypothesis = generate_hypothesis(abstract)
    similarity_score = compute_similarity(hypothesis, abstract)
    similarities.append(similarity_score)

    print(f"\n🧾 Abstract {i+1}:\n{abstract}")
    print(f"📌 Predicted Category: {predicted_category}")
    print(f"🧠 Generated Hypothesis: {hypothesis}")
    print(f"🔗 Similarity Score: {similarity_score:.4f}")

# Perform T-Test
similarities = np.array(similarities)
mean_sim = np.mean(similarities)
std_dev = np.std(similarities, ddof=1)
n = len(similarities)

t_stat, t_p = stats.ttest_1samp(similarities, popmean=0.7)

# Perform Z-Test
z_stat = (mean_sim - 0.7) / (std_dev / math.sqrt(n))
z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Print Summary
print("\n📊 Hypothesis Evaluation Summary")
print("---------------------------------")
print(f"Sample Size (n): {n}")
print(f"Mean Similarity: {mean_sim:.4f}")
print(f"Std Deviation:  {std_dev:.4f}")

print("\n🔬 T-Test:")
print(f"  T-Statistic = {t_stat:.4f}")
print(f"  P-Value     = {t_p:.4f}")
print("  ✅ Significant" if t_p < 0.05 else "  ❌ Not Significant")

print("\n🔬 Z-Test:")
print(f"  Z-Statistic = {z_stat:.4f}")
print(f"  P-Value     = {z_p:.4f}")
print("  ✅ Significant" if z_p < 0.05 else "  ❌ Not Significant")

# Compare tests
better = "T-Test" if t_p < z_p else "Z-Test"
print(f"\n🏆 More confident: {better} (lower p-value)")

# Visualize histogram
plt.figure(figsize=(8, 5))
sns.histplot(similarities, bins=10, kde=True, color="teal")
plt.title("Distribution of Hypothesis Similarity Scores")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
