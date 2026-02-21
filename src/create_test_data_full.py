##### code to convert the raw data (provided by organizers) to input prompt

import re
import os
import pandas as pd
from transformers import BertTokenizer



##### which one you want to convert
key = "test_phase_2"   ## "test_phase_2"  "valid"   "train"

import pandas as pd


# Read the necessary CSV files (the folder names used by organizers are same as key)
df_diagnosis = pd.read_csv('./'+key+'/'+'diagnosis.csv.gz', compression='gzip')
df_triage = pd.read_csv('./'+key+'/'+'triage.csv.gz', compression='gzip')
df_edstays = pd.read_csv('./'+key+'/'+'edstays.csv.gz', compression='gzip')

# Select the relevant columns from diagnosis and triage
df_diagnosis_subset = df_diagnosis[['stay_id', 'icd_title']]
df_triage_subset = df_triage[['stay_id', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'chiefcomplaint']]


df_merged = pd.merge(df_edstays, df_diagnosis_subset, on='stay_id', how='left')
df_merged = pd.merge(df_merged, df_triage_subset, on='stay_id', how='left')

df_merged = df_merged.groupby('hadm_id').agg(lambda x: ', '.join(x.astype(str))).reset_index()


columns_to_split = [
    "gender", "race", "arrival_transport", "disposition", "icd_title",
    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"
]

# Apply split and keep first element for each column
for col in columns_to_split:
    df_merged[col] = df_merged[col].astype(str).apply(lambda x: x.split(',')[0].strip())


import pandas as pd
df_rad = pd.read_csv('./'+key+'/'+'radiology.csv.gz', compression='gzip')

# Group by 'hadm_id' and concatenate the 'text' column
df_merged1 = df_rad.groupby('hadm_id')['text'].apply(lambda x: '\n================\n'.join(x)).reset_index()
df_merged1 = df_merged1.rename(columns={'text': 'radiology_text'})
# Save the merged dataframe to a new CSV file
df_merged = pd.merge(df_merged, df_merged1, on='hadm_id', how='left')


import pandas as pd
df_target = pd.read_csv('./'+key+'/'+'discharge_target.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
df_discharge = pd.read_csv('./'+key+'/'+'discharge.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')

# Keep only the specified columns
df_target = df_target[['hadm_id', 'discharge_instructions', 'brief_hospital_course']]
df_discharge = df_discharge[['hadm_id', 'text']]

# Merge the two dataframes on 'hadm_id'
merged_df = pd.merge(df_target, df_discharge, on='hadm_id', how='inner')

# Function to remove substrings from text
def remove_substrings(text, instructions, course):
    # Remove discharge instructions and brief hospital course from text
    if isinstance(instructions, str):
        text = text.replace(instructions, '', 1) # Remove only the first occurrence
    if isinstance(course, str):
        text = text.replace(course, '', 1) # Remove only the first occurrence
    return text.strip()

# Apply the function to create the cleaned text column
merged_df['cleaned_text'] = merged_df.apply(
    lambda row: remove_substrings(row['text'], row['discharge_instructions'], row['brief_hospital_course']),
    axis=1
)


merged_df = merged_df.drop(columns=['text'])
merged_df = merged_df.rename(columns={'cleaned_text': 'discharge_text'})


df_final_merged = pd.merge(df_merged, merged_df, on='hadm_id', how='left')










tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def count_bert_tokens(text):
    #return len((str(text)))
    return len(tokenizer.tokenize(str(text)))


def truncate_by_bert_tokens(text: str, max_tokens: int) -> str:
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
    return truncated_text

# Load your filtered data
filtered_df =  df_final_merged   #pd.read_csv(key+'_full.csv')



###### you can change the instructions if you wish

# Instructions
inst_bhc = """
You are a clinical language model. Below is the discharge note and radiology note from the MIMIC-IV dataset. 
Generate the "Brief Hospital Course" section, focusing only on:
- Clinical events
- Interventions and procedures
- Patient progress during admission
Exclude discharge instructions and follow-up care.
"""

inst_ds = """
You are a clinical language model. Below is the discharge note and radiology note from the MIMIC-IV dataset. 
Generate patient-facing discharge instructions based on both notes.
Include reason for admission, clinical events, interventions, discharge condition, and follow-up care.
"""

# Prompt template
alpaca_prompt = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{inputs}

### Response:
{outputs}
"""

# Function to format input string from structured data
def format_input(row):
    return (
        f"gender: {row['gender']}\n"
        f"arrival_transport: {row['arrival_transport']}\n"
        f"disposition: {row['disposition']}\n"
        f"icd_title: {row['icd_title']}\n"
        f"temperature: {row['temperature']}\n"
        f"heartrate: {row['heartrate']}\n"
        f"resprate: {row['resprate']}\n"
        f"o2sat: {row['o2sat']}\n"
        f"sbp: {row['sbp']}\n"
        f"dbp: {row['dbp']}\n"
        f"pain: {row['pain']}\n"
        f"acuity: {row['acuity']}\n"
        f"chiefcomplaint: {row['chiefcomplaint']}\n\n"
        f"## Radiology text\n: {truncate_by_bert_tokens(row.get('radiology_text', ''), 2500)}\n\n"
        f"## Discharge text\n: {truncate_by_bert_tokens(row.get('discharge_text', ''), 4000)}\n"
    )

# Generate BHC prompt examples
df_bhc = filtered_df.copy()
df_bhc["text"] = df_bhc.apply(
    lambda row: alpaca_prompt.format(
        instruction=inst_bhc.strip(),
        inputs=format_input(row),
        outputs=truncate_by_bert_tokens(row['brief_hospital_course'], 1000)  ###### , output will be '' for inference
    ),
    axis=1
)

# Generate DS prompt examples
df_ds = filtered_df.copy()
df_ds["text"] = df_ds.apply(
    lambda row: alpaca_prompt.format(
        instruction=inst_ds.strip(),
        inputs=format_input(row),
        outputs=truncate_by_bert_tokens(row['discharge_instructions'], 1000)  ##### , output will be '' for inference
    ),
    axis=1
)

# Combine the two DataFrames
df_combined = pd.concat([df_bhc, df_ds], ignore_index=True)



df = df_combined #pd.read_csv(key + '_full.csv')



df = df[['hadm_id', 'text', 'discharge_instructions', 'brief_hospital_course']]
# Save the filtered DataFrame
df.to_csv(key+'_full.csv', index=False)


print("Created and saved combined prompts with both BHC and Discharge Instructions.")

