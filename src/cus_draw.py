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
    """
    Process one Excel file to compute average comprehensive uniqueness scores
    for incremental 500-character segments across all documents.
    """
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

    '''avg_scores = []
    for idx in range(max_len):
        vals = [doc[idx] for doc in all_scores if idx < len(doc)]
        #print(vals)
        avg = sum(vals) / len(vals) if vals else None  # None if no snippet
        avg_scores.append(avg)'''
    
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
xlsx_files = glob.glob("./result_FT/"+key+"/*.xlsx")


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
















exit()



import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import glob
import os
import numpy as np
from statistics import mode, multimode
import statistics as stats

def compute_com_score(text):
    """
    Compute comprehensive uniqueness score for a given text.
    Uniqueness for n = unique_ngrams / total_ngrams, n in [3..10].
    Comprehensive score = product of all uniqueness scores.
    """
    text = re.sub(r'\s+', ' ', text.strip().lower())
    tokens = text.split()
    com_score = 1.0

    for n in range(3, 11):
        if len(tokens) < n:
            return 0
        ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        #print(ngrams)
        total = len(ngrams)
        unique_count = len(set(ngrams))
        #print(unique_count, total)
        uniqueness = unique_count / total
        com_score *= uniqueness
    #print("#########", com_score)
    return com_score


def process_file(file_path):
    """
    Process one Excel file to compute average comprehensive uniqueness scores
    for incremental 500-character segments across all documents.
    """
    df = pd.read_excel(file_path)

    texts = []
    if 'brief_hospital_course' in df.columns:
        texts.extend(df['brief_hospital_course'].dropna().astype(str).tolist())
    if 'discharge_instructions' in df.columns:
        texts.extend(df['discharge_instructions'].dropna().astype(str).tolist())

    all_scores = []
    for text in texts:
        scores = []
        k=0
        #print("$$$$$",len(text))
        #for start in range(500, len(text) + 500, 500):
        for start in range(500, 6000, 500):
            snippet = text[k:start]
            #print("@@@@@@", snippet)
            score = compute_com_score(snippet)
            scores.append(score)
            k=k+500
        all_scores.append(scores)
    #print("#####",all_scores)
    avg_len = sum(len(lst) for lst in all_scores) / len(all_scores)
    lengths = [len(lst) for lst in all_scores]
    mode_val = mode(lengths)
    median_val = np.median(lengths)
    quartiles = stats.quantiles(lengths, n=4)
    max_len = max(len(s) for s in all_scores)
    #max_len = int(quartiles[2])
    for s in all_scores:
        while len(s) < max_len:
            s.append(None)
    

    # Example: all_scores is a list of lists
    percentile_80 = np.percentile([len(lst) for lst in all_scores], 50)

    print(file_path, max_len, avg_len, percentile_80, mode_val, median_val, quartiles)
    avg_scores = []
    for idx in range(max_len):
        vals = [doc[idx] for doc in all_scores if doc[idx] is not None]
        avg = sum(vals) / len(vals) if vals else 0
        avg_scores.append(avg)

    return avg_scores


key = "fin_trunc_input"
# Path to XLSX files
xlsx_files = glob.glob("./"+key+"/*.xlsx")  # Change this path

plt.figure(figsize=(10, 6))

for file_path in xlsx_files:
    avg_scores = process_file(file_path)
    char_lengths = [(i+1)*500 for i in range(len(avg_scores))]
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    plt.plot(char_lengths, avg_scores, marker='o', label=file_name)

plt.xlabel("Number of Characters")
plt.ylabel("Average Comprehensive Uniqueness Score")
plt.title("Average Uniqueness Score vs Text Length")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save image to disk
plt.savefig("uniqueness_score_plot_"+key+".png", dpi=300)
plt.close()

print("Plot saved as 'uniqueness_score_plot.png'")











exit()




import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import glob
import os

def compute_uniqueness_per_n(text):
    """
    Compute uniqueness score for each n in [3..10] for a given text.
    Uniqueness = unique_ngrams / total_ngrams.
    """
    text = re.sub(r'\s+', ' ', text.strip().lower())
    tokens = text.split()
    scores = []
    for n in range(3, 11):
        if len(tokens) < n:
            scores.append(None)
        else:
            ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            total = len(ngrams)
            unique_count = len(set(ngrams))
            scores.append(unique_count / total)
    return scores

def process_file(file_path):
    """
    Process one Excel file to compute average comprehensive uniqueness scores
    for incremental 500-character segments across all documents.
    Averages are computed only from documents that have data for that length.
    """
    df = pd.read_excel(file_path)

    texts = []
    if 'brief_hospital_course' in df.columns:
        texts.extend(df['brief_hospital_course'].dropna().astype(str)
                     .apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
                     .tolist())
    if 'discharge_instructions' in df.columns:
        texts.extend(df['discharge_instructions'].dropna().astype(str)
                     .apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
                     .tolist())

    all_scores = []  # list of lists, each inner list: com_score for each length
    for text in texts:
        segment_scores = []
        for start in range(500, len(text) + 500, 500):
            snippet = text[:start]
            n_scores = compute_uniqueness_per_n(snippet)
            # Multiply only non-None scores
            valid_scores = [s for s in n_scores if s is not None]
            if valid_scores:
                com_score = 1
                for s in valid_scores:
                    com_score *= s
                segment_scores.append(com_score)
            else:
                segment_scores.append(None)
        all_scores.append(segment_scores)

    max_len = max(len(s) for s in all_scores)
    avg_scores = []
    for idx in range(max_len):
        vals = [doc[idx] for doc in all_scores if doc[idx] is not None]
        avg_scores.append(sum(vals) / len(vals) if vals else None)

    return avg_scores

# Path to XLSX files
xlsx_files = glob.glob("./input/*.xlsx")

plt.figure(figsize=(10, 6))

for file_path in xlsx_files:
    avg_scores = process_file(file_path)
    char_lengths = [(i+1)*500 for i in range(len(avg_scores))]
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    plt.plot(char_lengths, avg_scores, marker='o', label=file_name)

plt.xlabel("Number of Characters")
plt.ylabel("Average Comprehensive Uniqueness Score")
plt.title("Average Uniqueness Score vs Text Length")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("uniqueness_score_plot.png", dpi=300)
plt.close()
print("Plot saved as 'uniqueness_score_plot.png'")





exit()
