import regex as re
from tqdm import tqdm

def clean_wiki_text(content):
    """Clean Wikipedia text content following wikifil.pl conventions"""
    # Remove redirect text content
    txt = re.sub('^#REDIRECT.*$', '', content)
    
    # Remove references with their content
    txt = re.sub(r'<ref[^<]*<\/ref>', '', txt)
    
    # Remove various wiki markup
    replacements = [
        (r'\{\{.*?\}\}', ''),  # Remove templates
        (r'\{\|.*?\|\}', ''),  # Remove tables
        (r"'{2,5}", ''),       # Remove bold/italic markers
        (r'\[\[(Category|Image|Media|File):.*?\]\]', ''),  # Remove media/category links
        (r'\|thumb\b|\|left\b|\|right\b|\|\d+px\b', ''),  # Remove image options
        (r'\[\[[a-z\-]*:[^\]]*\]\]', ''),  # Remove interlanguage links
        (r'\[\[[^\|\]]*\|', '[['),  # Clean wiki links, preserve visible text
        (r'\[\[|\]\]', ''),    # Remove remaining [[ and ]]
        (r'\[http:[^] ]*', '['),  # Clean URLs, preserve visible text
        (r'\[|\]', ''),        # Remove remaining [ and ]
        (r'={2,5}.*?={2,5}', ''),  # Remove headers
        (r'&lt;|&gt;|&quot;|&amp;|&nbsp;', ' '),  # Convert HTML entities
        (r'<[^>]*>', ''),      # Remove remaining HTML tags
        (r'[*#:;]+\s.*?\n', '\n'),  # Remove list items and indents
        (r'(\n\s*){3,}', '\n\n'),  # Normalize multiple newlines
        (r'^\s+|\s+$', '')     # Trim whitespace
    ]
    
    for pattern, replacement in replacements:
        txt = re.sub(pattern, replacement, txt, flags=re.MULTILINE|re.DOTALL)
    
    # Add this new section to filter characters
    allowed_chars = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ⏎'''
    txt = ''.join(c for c in txt if c in allowed_chars)
    
    return txt

if __name__ == "__main__":
    
    INPUT_PATH = "data/enwiki/enwik9"
    OUTPUT_PATH = "data/enwiki/enwik9_clean.txt"
    f1 = open(INPUT_PATH, 'r', encoding='utf-8').read()
    matches = re.findall('<text.*?>(.*?)</text>', f1, flags=re.S)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fw:
        for content in tqdm(matches, total=len(matches), desc="Cleaning wiki dataset"):
            clean_txt = clean_wiki_text(content)
            fw.write(clean_txt + '\n')
