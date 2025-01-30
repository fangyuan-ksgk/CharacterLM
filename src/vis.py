import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects
import matplotlib.gridspec as gridspec

# (b). Add color to each word within the text 
#    - add color to the first word in the first row 
def get_full_text(text, char_per_row=60): 
    num_rows = 1
    full_text = '' 
    row_text = ''
    for word in text.split(): 
        if len(row_text) + len(word) > char_per_row: 
            if full_text == '':
                full_text = row_text 
            else: 
                full_text += '\n' + row_text
            row_text = word
            num_rows += 1
        else: 
            if row_text == "": 
                row_text = word
            else: 
                row_text += " " + word
    full_text += "\n" + row_text
    return full_text, num_rows 

def display_colored_text(text, char_colors, title=None, char_per_row=60):
    """
    Display text with individual character coloring
    """
    plt.figure(figsize=(15, 5))
    if title: 
        plt.title(title, fontsize=30, fontweight='bold')
    
    # Split text into lines
    full_text, num_rows = get_full_text(text, char_per_row)
    lines = full_text.split('\n')
    
    y_pos = 0.90
    line_height = 0.15
    char_width = 0.015
    for line_num, line in enumerate(lines):
        x_pos = 0.05
        current_text = ""
        
        for i, char in enumerate(line):
            orig_idx = sum(len(l) for l in lines[:line_num]) + i
            if orig_idx < len(char_colors):
                color = char_colors[orig_idx]
                
                if color:
                    # Print accumulated non-highlighted text
                    if current_text:
                        plt.text(x_pos, y_pos, current_text,
                                fontfamily='monospace',
                                va='top',
                                fontsize=24)
                        x_pos += len(current_text) * char_width
                        current_text = ""
                    
                    # Print highlighted character without extra spacing
                    plt.text(x_pos, y_pos, char,
                            fontfamily='monospace',
                            va='top',
                            fontsize=24,
                            backgroundcolor=color,
                            bbox=dict(pad=0.0, facecolor=color, edgecolor='none'))
                    x_pos += char_width * 1.0 # slightly wider due to background color box
                else:
                    current_text += char
            
        # Print any remaining non-highlighted text
        if current_text:
            plt.text(x_pos, y_pos, current_text,
                    fontfamily='monospace',
                    va='top',
                    fontsize=24)
        
        y_pos -= line_height
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    
def display_colored_text_with_histogram(text, char_colors, title=None, char_per_row=60):
    """
    Display text with individual character coloring with perplexity histograms
    """
    # Split text into lines
    full_text, num_rows = get_full_text(text, char_per_row)
    lines = full_text.split('\n')
    
    # Create figure with gridspec for precise control
    fig = plt.figure(figsize=(15, max(5, num_rows * 1.5)))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
    
    # Histogram subplot
    ax1 = plt.subplot(gs[0])
    ax1.hist(char_colors, bins=50, color='blue', alpha=0.6)
    ax1.set_xlim(0, 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Text subplot
    ax2 = plt.subplot(gs[1])
    
    y_pos = 0.90
    line_height = 0.15
    char_width = 0.015
    hist_height = 0.03  # Height of histogram
    
    for line_num, line in enumerate(lines):
        x_pos = 0.05
        current_text = ""
        
        # Calculate perplexity values for this line
        line_start_idx = sum(len(l) for l in lines[:line_num])
        line_colors = char_colors[line_start_idx:line_start_idx + len(line)]
        perplexity_values = [c[3] if isinstance(c, tuple) else 0 for c in line_colors if c]
        
        # Draw histogram above the line
        if perplexity_values:
            ax_hist = plt.axes([0.05, y_pos + 0.02, 0.9, hist_height])
            ax_hist.hist(perplexity_values, bins=20, color='gray', alpha=0.5)
            ax_hist.set_xticks([])
            ax_hist.set_yticks([])
            for spine in ax_hist.spines.values():
                spine.set_visible(False)
        
        # Rest of the text rendering
        for i, char in enumerate(line):
            orig_idx = sum(len(l) for l in lines[:line_num]) + i
            if orig_idx < len(char_colors):
                color = char_colors[orig_idx]
                
                if color:
                    # Print accumulated non-highlighted text
                    if current_text:
                        plt.text(x_pos, y_pos, current_text,
                                fontfamily='monospace',
                                va='top',
                                fontsize=24)
                        x_pos += len(current_text) * char_width
                        current_text = ""
                    
                    # Print highlighted character without extra spacing
                    plt.text(x_pos, y_pos, char,
                            fontfamily='monospace',
                            va='top',
                            fontsize=24,
                            backgroundcolor=color,
                            bbox=dict(pad=0.0, facecolor=color, edgecolor='none'))
                    x_pos += char_width * 1.0
                else:
                    current_text += char
            
        # Print any remaining non-highlighted text
        if current_text:
            plt.text(x_pos, y_pos, current_text,
                    fontfamily='monospace',
                    va='top',
                    fontsize=24)
        
        y_pos -= line_height
    
    ax2.axis('off')
    
    # Adjust spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2)
    
    if title:
        fig.suptitle(title, fontsize=30, fontweight='bold', y=0.98)
    
    plt.show()