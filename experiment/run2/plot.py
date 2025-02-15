import pickle
import glob 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os 
import numpy as np

pkl_dir = "experiment/run2"

def plot_bpc_vs_vocab_size(pkl_dir, 
                           log_scale_vocab = False,
                           increase_vocab_size = False,
                           title: str = "BPC vs Vocab Size"):
    pkl_path = pkl_dir + "/*.pkl"
    vocab_sizes = []
    bpcs = []
    for pkl_path in glob.glob(pkl_path):
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

    # Sort values based on increase_vocab_size parameter
    df = df.sort_values('Vocabulary Size', ascending=True)

    if log_scale_vocab:
        # Transform the x values to log scale for fitting
        df['Log Vocabulary Size'] = np.log10(df['Vocabulary Size'])
        
        plt.figure(figsize=(10, 6))
        # Use the sorted order for both scatter and regression
        ax = sns.scatterplot(data=df, x='Log Vocabulary Size', y='Bits per Character', alpha=0.6)
        sns.regplot(data=df, x='Log Vocabulary Size', y='Bits per Character', 
                   scatter=False, color='red')
        plt.xscale('log')
        if not increase_vocab_size:
            plt.gca().invert_xaxis()
    else:
        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(data=df, x='Vocabulary Size', y='Bits per Character', alpha=0.6)
        sns.regplot(data=df, x='Vocabulary Size', y='Bits per Character', 
                   scatter=False, color='red')
        if not increase_vocab_size:
            plt.gca().invert_xaxis()

    # Customize the plot
    plt.title(title, pad=15)
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Bits per Character (BPC)')

    min_idx = df['Vocabulary Size'].idxmin() if increase_vocab_size else df['Vocabulary Size'].idxmax()
    max_idx = df['Vocabulary Size'].idxmax() if increase_vocab_size else df['Vocabulary Size'].idxmin()
    
    # Convert log values back to actual vocabulary sizes
    x_col = 'Vocabulary Size'  # Always use actual vocabulary size for plotting
    min_x = df[x_col].iloc[min_idx]
    max_x = df[x_col].iloc[max_idx]
    
    # Add annotations for each point
    shift_magnitude = 80  # Define the magnitude of the shift once
    for idx, row in df.iterrows():
        if idx != min_idx and idx != max_idx: 
            continue
        if idx == min_idx: 
            xyshift = (shift_magnitude, 0)  # Right shift
            anno_text = "Min Vocab"
        else: 
            xyshift = (-shift_magnitude * 2, 0)  # Left shift with same magnitude
            anno_text = "Max Vocab"
            
        x = row['Vocabulary Size' if not log_scale_vocab else 'Log Vocabulary Size']
        y = row['Bits per Character']
        plt.annotate(f'{anno_text} BPC: {y:.4f}',
                    xy=(x, y),
                    xytext=xyshift,  # Offset more to the right
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                    arrowprops=dict(arrowstyle='<-'))

    plt.tight_layout()