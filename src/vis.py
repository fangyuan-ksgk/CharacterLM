import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects

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

def display_colored_text(text, char_colors, char_per_row=60):
    """
    Display text with individual character coloring
    """
    plt.figure(figsize=(15, 5))
    
    # Split text into lines
    full_text, num_rows = get_full_text(text, char_per_row)
    lines = full_text.split('\n')
    
    y_pos = 0.95
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