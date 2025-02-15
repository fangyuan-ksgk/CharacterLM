from util import clean_wiki_text
import re
from tqdm import tqdm

if __name__ == "__main__":
    
    INPUT_PATH = "data/enwiki/enwik8"
    OUTPUT_PATH = "data/enwiki/enwik8_clean.txt"
    f1 = open(INPUT_PATH, 'r', encoding='utf-8').read()
    matches = re.findall('<text.*?>(.*?)</text>', f1, flags=re.S)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fw:
        for content in tqdm(matches, total=len(matches), desc="Cleaning wiki dataset"):
            clean_txt = clean_wiki_text(content)
            fw.write(clean_txt + '\n')