import numpy as np
import matplotlib.patheffects
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.path import Path


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
    
    

from typing import Optional

def visualize_text_multiline(text, char_colors, groups: Optional[list]=None, 
                             output_path: Optional[str] = None, 
                             max_chars_per_row=10, 
                             title = None,
                             fontsize=24,
                             row_gap=1,
                             const=1.0):
    """
    Visualizes text wrapped into multiple rows with per-character background colors 
    and annotated bracket groups.

    Parameters:
      text (str): The text to display.
      char_colors (list of str): A list of background colors (one per character).
      groups (list of tuples): Each tuple may be:
             (start, end)
             (start, end, label)
             (start, end, label, bracket_color)
             Here, start and end are indices into the full text (end is exclusive).
             Groups spanning multiple rows are automatically split.
      max_chars_per_row (int): Maximum number of characters to display per row.
    """
    rows = [text[i:i+max_chars_per_row] for i in range(0, len(text), max_chars_per_row)]
    color_rows = [char_colors[i:i+max_chars_per_row] for i in range(0, len(text), max_chars_per_row)]
    n_rows = len(rows)

    if groups:
        row_gap += 0.5  # Adjust as needed

    fig, ax = plt.subplots(figsize=(max(8, max_chars_per_row * 0.3), row_gap * n_rows))

    scaled_row_gap = row_gap * (fontsize / 24) 
    for row_idx, (row_text, row_colors) in enumerate(zip(rows, color_rows)):

        y = -row_idx * scaled_row_gap
        for col_idx, (ch, bg_color) in enumerate(zip(row_text, row_colors)):
            rect_height = fontsize/36
            rect_width = fontsize/36 * 1.3
            rect = patches.Rectangle((col_idx - rect_width/2, y - rect_height/2), rect_width, rect_height,
                                     facecolor=bg_color,
                                     edgecolor='none',
                                     zorder=2)
            ax.add_patch(rect)
            
            ax.text(col_idx, y, ch,
                    fontsize=fontsize,
                    fontfamily='monospace',
                    ha='center', va='center',
                    zorder=2)

    if groups:
        for group in groups:
            # Unpack the group tuple.
            if len(group) == 4:
                start, end, label, bracket_color = group
            elif len(group) == 3:
                start, end, label = group
                bracket_color = 'black'
            else:
                start, end = group
                label = None
                bracket_color = 'black'
            
            first_row = start // max_chars_per_row
            last_row  = (end - 1) // max_chars_per_row  # end is exclusive

            for row in range(first_row, last_row + 1):
                row_start_index = row * max_chars_per_row
                row_end_index   = row_start_index + max_chars_per_row
                seg_start = max(start, row_start_index)
                seg_end   = min(end, row_end_index)
                if seg_end <= seg_start:
                    continue  # No characters in this row for the group

                local_start = seg_start - row_start_index
                local_end   = seg_end - row_start_index  # this is exclusive

                x_left  = local_start - 0.3
                x_right = local_end - 0.7

                y_text = -row * row_gap
                y_top    = y_text - 0.4
                y_bottom = y_text - 0.7

                offset = (y_top - y_bottom)  # Here, if y_top-y_bottom=0.3, offset=0.3

                # Define vertices for the curved bracket:
                vertices = [
                    (x_left, y_top),                       # Start at top left
                    (x_left - offset/2, y_top),              # Control point for left curve
                    (x_left - offset/2, y_bottom),           # Second control point for left curve
                    (x_left + offset, y_bottom),                    # Left-bottom point
                    (x_right - offset, y_bottom),                   # Right-bottom point (straight segment)
                    (x_right + offset/2, y_bottom),          # Control point for right curve
                    (x_right + offset/2, y_top),             # Second control point for right curve
                    (x_right, y_top)                       # End at top right
                ]
                codes = [
                    Path.MOVETO,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.LINETO,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CURVE4
                ]
                bracket_path = Path(vertices, codes)
                bracket_patch = patches.PathPatch(bracket_path,
                                                fill=False,
                                                lw=2,
                                                color=bracket_color,
                                                zorder=0)
                ax.add_patch(bracket_patch)
                
                # Place the label only on the first segment if provided.
                if label and row == first_row:
                    ax.text((x_left + x_right) / 2, y_bottom - 0.1, label,
                            ha='center', va='top', fontsize=13, zorder=2, color=bracket_color)
    
    ax.set_xlim(-1, max_chars_per_row)
    ax.set_ylim(-row_gap * n_rows - fontsize/72 + 0.5, 1)
    ax.axis('off')
    if title: 
        plt.title(title, fontsize=30, fontweight='bold')
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)
    
    return 