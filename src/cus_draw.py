# run the comprehensive uniqueness score code, it also draw a graph too


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import glob
import os
import numpy as np
from statistics import mode
import statistics as stats

q = 200

def compute_com_score(text):
    text = re.sub(r'\s+', ' ', text.strip().lower())
    tokens = text.split()
    com_score = 1.0

    for n in range(3, 11):
        if len(tokens) < n:
            return 0
        ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        total = len(ngrams)
        unique_count = len(set(ngrams))
        uniqueness = unique_count / total
        com_score *= uniqueness

    return com_score


def process_file(file_path):
    df = pd.read_excel(file_path)

    texts = []
    if 'brief_hospital_course' in df.columns:
        texts.extend(df['brief_hospital_course'].dropna().astype(str).tolist())
    if 'discharge_instructions' in df.columns:
        texts.extend(df['discharge_instructions'].dropna().astype(str).tolist())

    all_scores = []
    ks=[]
    for text in texts:
        scores = []
        k = 0
        #for start in range(500, len(text) + 500, 500):
        for start in range(500, 2500, q):
            if len(text) < start:  
                break  # stop if snippet is longer than text
            snippet = text[k:start]
            #snippet = text[0:start]
            #snippet = text[start-500:start]
            score = compute_com_score(snippet)
            scores.append(score)
            ks.append(k)
            k += q
        all_scores.append(scores)

    max_len = max(len(s) for s in all_scores) if all_scores else 0
    
    avg_scores = []
    for idx in range(max_len):
        # take only non-zero values
        vals = [doc[idx] for doc in all_scores if idx < len(doc) and doc[idx] != 0]
        avg = sum(vals) / len(vals) if vals else None  # None if all values were 0
        avg_scores.append(avg)

    print(avg_scores)
    # Trim trailing None values (no docs at that length)
    while avg_scores and (avg_scores[-1] is None or avg_scores[-1] <0.1):
        #print(avg_scores[-1])
        avg_scores.pop()

    print(ks)
    return avg_scores


import os
key = "Full"
xlsx_files = glob.glob("./path_to_your_output_files/"+key+"/*.xlsx")  #### the files which we get from, out_to_score_inp.py or reshape.py


model_names = [os.path.splitext(os.path.basename(p))[0].split(" ")[0].strip() for p in xlsx_files]

print(xlsx_files)
plt.figure(figsize=(10, 6))



for i in range(len(xlsx_files)):
    avg_scores = process_file(xlsx_files[i])
    print(avg_scores)
    char_lengths = [500+(i)*q for i in range(len(avg_scores))]
    file_name = os.path.splitext(os.path.basename(xlsx_files[i]))[0]

    if avg_scores:  # Only plot if non-empty
        plt.plot(char_lengths, avg_scores, marker='o', label=model_names[i])

plt.xlabel("Number of Characters")
plt.ylabel("Average Comprehensive Uniqueness Score")
plt.title("Average Uniqueness Score vs Text Length")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("uniqueness_score_plot_"+key.replace("/","_")+"_FT.png", dpi=300)
plt.show()
plt.close()


print("Plot saved as 'uniqueness_score_plot_"+key.replace("/","_")+".png'")



