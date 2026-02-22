##### keep this file in the folder where the python codes for scoring calculation are there for calculate evaluation score


import os
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

### all scoring modules kept same as organizer's provided code 
from bertscore import BertScore
from rouge import Rouge
from bleu import Bleu
import evaluate
from align import AlignScorer
from UMLSScorer import UMLSScorer

# Paths
PATH = "./path_to_input_files/"  ###### input path same as the original codes provided by organizers
RES_PATH = os.path.join(PATH, "path_to_generated_files_after_reshape") # same as 'res' folder in provided code, we use this folder to keep our generated files in xlsx format
REF_PATH = os.path.join(PATH, "path_to_reference_data/target_A.csv") # same as reference dataset, we generate our own reference dataset using organizers provided 250 samples 

# Load reference once
reference = pd.read_csv(REF_PATH, keep_default_na=False)
reference["hadm_id"] = reference["hadm_id"].astype(int)
reference = reference.sort_values(by="hadm_id")

refsdi = reference["discharge_instructions"].tolist()
refsbhc = reference["brief_hospital_course"].tolist()

# Initialize scorers once
rougeScorer = Rouge(["rouge1", "rouge2", "rougeL"])
bleuScorer = Bleu()
meteorScorer = evaluate.load("meteor")
alignScorer = AlignScorer()
medconScorer = UMLSScorer(quickumls_fp="./quickumls/")
bertScorer = BertScore()


def calculate_scores(generated, reference, metrics):
    scores = {}
    for metric in metrics:
        scores[metric] = {"discharge_instructions": [], "brief_hospital_course": []}

    if "medcon" in metrics:
        medconScorer = UMLSScorer(quickumls_fp="./quickumls/")

    def process(rows_ref, rows_gen):
        if "medcon" in metrics:
            temp = medconScorer(
                rows_ref["discharge_instructions"].tolist(),
                rows_gen["discharge_instructions"].tolist(),
            )
            scores["medcon"]["discharge_instructions"].append(temp)

            temp = medconScorer(
                rows_ref["brief_hospital_course"].tolist(),
                rows_gen["brief_hospital_course"].tolist(),
            )
            scores["medcon"]["brief_hospital_course"].append(temp)

    reference.set_index("hadm_id", drop=False, inplace=True)
    generated.set_index("hadm_id", drop=False, inplace=True)

    batch_size = 8
    for i in range(0, len(generated), batch_size):
        rows_ref = reference[i : i + batch_size]
        rows_gen = generated[i : i + batch_size]
        process(rows_ref, rows_gen)

    return scores


def compute_overall_score(scores):
    leaderboard = {}
    if "medcon" in scores:
        medcon_discharge_instructions = np.mean(scores["medcon"]["discharge_instructions"])
        medcon_brief_hospital_course = np.mean(scores["medcon"]["brief_hospital_course"])
        leaderboard["medcon"] = np.mean([medcon_discharge_instructions, medcon_brief_hospital_course])
    overall_score = np.mean(list(leaderboard.values()))
    return medcon_discharge_instructions, medcon_brief_hospital_course, overall_score



def run_file_specific(file):
    if not file.endswith(".xlsx"):
        return 0

    key = os.path.splitext(file)[0]   
    print(f"\n=== Processing {key} ===")

    generated = pd.read_excel(os.path.join(RES_PATH, file), keep_default_na=False)
    generated["hadm_id"] = generated["hadm_id"].astype(int)
    generated = generated.sort_values(by="hadm_id")

    hypsdi = generated["discharge_instructions"].tolist()
    hypsbhc = generated["brief_hospital_course"].tolist()

    # Open output CSV
    out_csv = key + "___processor.csv"
    with open(out_csv, "w") as fil:
        columns = ['char_len', 'BERTScore_DI','BERTScore_BHC','BERTScore_AVG',
                   'MEDCON_DI','MEDCON_BHC','MEDCON_AVG',
                   'ALIGN_DI','ALIGN_BHC','ALIGN_AVG',
                   'METEOR_DI','METEOR_BHC','METEOR_AVG',
                   'BLUE_DI','BLUE_BHC','BLUE_AVG',
                   'ROUGE_1_DI','ROUGE_1_BHC','ROUGE_1_AVG',
                   'ROUGE_2_DI','ROUGE_2_BHC','ROUGE_2_AVG',
                   'ROUGE_L_DI','ROUGE_L_BHC','ROUGE_L_AVG']
        fil.write(",".join(columns) + "\n")

        k = 800
        #while k < 1100:
        print(k, key)
        gc.collect()
        k += 200
        fil.write(str(k)+",")

        hypsdi1 = [x for x in hypsdi]
        hypsbhc1 = [x for x in hypsbhc]

        # === BERTScore ===
        di_score = np.mean(bertScorer(refsdi, hypsdi1))
        bhc_score = np.mean(bertScorer(refsbhc, hypsbhc1))
        avg_score = (di_score + bhc_score) / 2.0
        fil.write(f"{di_score},{bhc_score},{avg_score},")
        # === MEDCON ===
        gen_trunc = generated.copy()
        for col in ["discharge_instructions", "brief_hospital_course"]:
            gen_trunc[col] = gen_trunc[col].str[:k]
        scores = calculate_scores(gen_trunc, reference, metrics=["medcon"])
        di, bhc, avg = compute_overall_score(scores)
        fil.write(f"{di},{bhc},{avg},")
        # === ALIGN ===
        di_score = np.mean([alignScorer(refsdi, hypsdi1)])
        bhc_score = np.mean([alignScorer(refsbhc, hypsbhc1)])
        avg_score = (di_score + bhc_score) / 2.0
        fil.write(f"{di_score},{bhc_score},{avg_score},")

        # === METEOR ===
        di_score = np.mean(meteorScorer.compute(references=refsdi, predictions=hypsdi1)["meteor"])
        bhc_score = np.mean(meteorScorer.compute(references=refsbhc, predictions=hypsbhc1)["meteor"])
        avg_score = (di_score + bhc_score) / 2.0
        fil.write(f"{di_score},{bhc_score},{avg_score},")

        # === BLEU ===
        di_score = np.mean([bleuScorer(refsdi, hypsdi1)])
        bhc_score = np.mean([bleuScorer(refsbhc, hypsbhc1)])
        avg_score = (di_score + bhc_score) / 2.0
        fil.write(f"{di_score},{bhc_score},{avg_score},")

        # === ROUGE ===
        di_rouge = rougeScorer(refsdi, hypsdi1)
        bhc_rouge = rougeScorer(refsbhc, hypsbhc1)
        fil.write(f"{np.mean(di_rouge['rouge1'])},{np.mean(bhc_rouge['rouge1'])},{(np.mean(di_rouge['rouge1'])+np.mean(bhc_rouge['rouge1']))/2.0},")
        fil.write(f"{np.mean(di_rouge['rouge2'])},{np.mean(bhc_rouge['rouge2'])},{(np.mean(di_rouge['rouge2'])+np.mean(bhc_rouge['rouge2']))/2.0},")
        fil.write(f"{np.mean(di_rouge['rougeL'])},{np.mean(bhc_rouge['rougeL'])},{(np.mean(di_rouge['rougeL'])+np.mean(bhc_rouge['rougeL']))/2.0}\n")

        fil.flush()

    print(f"Finished {key}, results saved to {out_csv}")
    return 1









##################  CPU, multi-processing

from multiprocessing import Pool, cpu_count
import os
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    files = os.listdir(RES_PATH)

    with ProcessPoolExecutor() as executor:
        executor.map(run_file_specific, files)





