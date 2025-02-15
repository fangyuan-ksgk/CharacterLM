import pickle
import glob 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
pkl_paths = "checkpoint/run1/*.pkl"
vocab_sizes = []
bpcs = []
for pkl_path in glob.glob(pkl_paths):
    with open(pkl_path, "rb") as f:
        info = pickle.load(f)
        vocab_size = info["config"]["vocab_size"]
        bpc = info["bpc"].item()
    vocab_sizes.append(vocab_size)
    bpcs.append(bpc)

# Create a DataFrame for better plotting
df = pd.DataFrame({
    'Vocabulary Size': vocab_sizes,
    'Bits per Character': bpcs
})

# Set the style
sns.set_style("whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Vocabulary Size', y='Bits per Character', alpha=0.6)
sns.regplot(data=df, x='Vocabulary Size', y='Bits per Character', scatter=False, color='red')

# Customize the plot
plt.title('BPC vs Vocab Size', pad=15)
plt.xlabel('Vocabulary Size')
plt.ylabel('Bits per Character (BPC)')

# Add annotations
plt.annotate(f'CharacterVocab Min BPC: {min(bpcs):.4f}', 
            xy=(vocab_sizes[bpcs.index(min(bpcs))], min(bpcs)),
            xytext=(10, 10), textcoords='offset points')

plt.tight_layout()