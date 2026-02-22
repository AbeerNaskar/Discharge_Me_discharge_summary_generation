# zero shot : generate bhc and di from discharge (truncated) and structured info 




import time
import math


def format_seconds(seconds):
    # Use math.modf to split fractional seconds (kept) from total seconds (used for D/H/M)
    fractional, total_seconds = math.modf(seconds)
    total_seconds = int(total_seconds)
    
    # Calculate components
    days, total_seconds = divmod(total_seconds, 86400)
    hours, total_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(total_seconds, 60)
    
    # Format the output string, including milliseconds
    time_str = f"{seconds:02d}.{int(fractional * 1000):03d}s"
    if minutes > 0:
        time_str = f"{minutes:02d}m {time_str}"
    if hours > 0:
        time_str = f"{hours:02d}h {time_str}"
    if days > 0:
        time_str = f"{days}d {time_str}"
        
    return time_str




start_time = time.perf_counter()
print(f"[{format_seconds(0)}] Script started.")


print("gemma3 full predict zero")









import os, torch, pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel  # Unsloth's high‑level wrappers



max_seq_length = 8000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.




# provide the instruct tuned model path
model, tokenizer = FastLanguageModel.from_pretrained(  ### for Gemma-3 model use ; from unsloth import FastModel 
    model_name = "/path_to_model/gemma-2-9b-it-bnb-4bit", # other unsloth models: "phi-4-unsloth-bnb-4bit", "mistral-7b-instruct-v0.3-bnb-4bit", "Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "", # if required put your huggingface token
)






import gc
gc.collect()


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

flush()





end_part1 = time.perf_counter()
elapsed_part1 = end_part1 - start_time
print(f"[{format_seconds(elapsed_part1)}] model loaded.")
elapsed_part1 = end_part1







del_lst = ["<end_of_turn>", "<eos>", " <bos>"] 


def del_trash(st):
    for t in del_lst:
        st = st.replace(t,"")
    return st.strip()


prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{question}

### Input:
{note}

### Response:
"""






# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference





def summerize(note, question):
    inp = prompt.format(note=note, question=question)
    inputs = tokenizer([inp], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 1000, use_cache = True, repetition_penalty=1.2, top_k=50, top_p=0.95)
    out = tokenizer.batch_decode(outputs)[0].replace(inp, "").replace("<|begin_of_text|>","").replace("<|end_of_text|>","").replace("<eos>","").replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "").replace("<end_of_turn>", "").replace("<bos>", "").replace("<eos>", "").replace("<|eot_id|>", "").replace("<s>", "")
    return out 




def truncated_to_limit(st, limit=2500):
    token = tokenizer(st, truncation=True, max_length=limit, return_tensors="pt")
    tokens = token['input_ids'][0]
    del token
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return tokenizer.decode(tokens, skip_special_tokens=True)







##### which one you want to convert
from transformers import BertTokenizer
key = "test_phase_2"  


import pandas as pd
# Read the necessary CSV files (the folder names used by organizers are same as key)
df_diagnosis = pd.read_csv('./'+key+'/'+'diagnosis.csv.gz', compression='gzip')
df_triage = pd.read_csv('./'+key+'/'+'triage.csv.gz', compression='gzip')
df_edstays = pd.read_csv('./'+key+'/'+'edstays.csv.gz', compression='gzip')

# Select the relevant columns from diagnosis and triage
df_diagnosis_subset = df_diagnosis[['stay_id', 'icd_title']]
df_triage_subset = df_triage[['stay_id', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'chiefcomplaint']]

# Merge the dataframes based on the common column 'stay_id'
# Start with df_edstays as the base dataframe to keep all its rows
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



# prompt: read, radiology.csv.gz, here under the 'text' column there are multiple entries, corresponding to 'hadm_id' column, merge those and output in new csv file

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
    # Remove the specified phrases
    return text.strip()

# Apply the function to create the cleaned text column
merged_df['cleaned_text'] = merged_df.apply(
    lambda row: remove_substrings(row['text'], row['discharge_instructions'], row['brief_hospital_course']),
    axis=1
)


merged_df = merged_df.drop(columns=['text'])
merged_df = merged_df.rename(columns={'cleaned_text': 'discharge_text'})



df_final_merged = pd.merge(df_merged, merged_df, on='hadm_id', how='left')




df = df_final_merged

# Predefined section headers
section_headers = [
    "Allergies:", "Attending:", "Chief Complaint:", "Major Surgical or Invasive Procedure:",
    "History of Present Illness:", "Past Medical History:", "Social History:", "Family History:",
    "Physical Exam:", "Pertinent Results:", "Brief Hospital Course:", "Medications on Admission:",
    "Discharge Medications:", "Discharge Disposition:", "Discharge Diagnosis:",
    "Discharge Condition:", "Discharge Instructions:", "Followup Instructions:"
]

# Escape special characters and make regex pattern
header_pattern = '|'.join(map(re.escape, section_headers))

# Function to split text into sections
def split_into_sections(text):
    section_dict = {}
    text = str(text).replace("___", "").strip()

    # Use regex to find all header locations
    matches = list(re.finditer(header_pattern, text))
    
    # Handle text before first section
    if matches:
        start_text = text[:matches[0].start()].strip()
        if start_text:
            section_dict['start'] = start_text

        for i, match in enumerate(matches):
            header = match.group()
            start_idx = match.end()
            end_idx = matches[i+1].start() if i+1 < len(matches) else len(text)
            section_body = text[start_idx:end_idx].strip().replace("___", "")
            section_dict[header.rstrip(':')] = section_body
    else:
        section_dict['start'] = text  # if no headers found

    return section_dict


# Apply section extraction
df['section_dict'] = df['discharge_text'].apply(split_into_sections)

# Formatting function
def format_dict_output(data: dict) -> str:
    output_lines = []
    for key, value in data.items():
        cleaned_value = str(value).strip()
        output_lines.append(f"{key}: {cleaned_value}\n")
    return '\n'.join(output_lines)

# Function to build BHC_discharge
def build_bhc_discharge(sec_dict):
    keys_to_keep = [
        "Allergies", "Chief Complaint", "Major Surgical or Invasive Procedure", 
        "History of Present Illness", "Past Medical History", "Social History", 
        "Family History", "Physical Exam", "Pertinent Results"
    ]
    filtered = {}
    if 'start' in sec_dict:
        start_lines = sec_dict['start'].splitlines()
        for line in start_lines:
            if "Service:" in line:
                filtered['start'] = line.strip()
                break
    for key in keys_to_keep:
        if key in sec_dict:
            filtered[key] = sec_dict[key]
    return format_dict_output(filtered)

# Function to build DI_discharge
def build_di_discharge(sec_dict):
    keys_to_keep = [
        "Chief Complaint", "History of Present Illness", "Physical Exam",
        "Medications on Admission", "Discharge Medications", "Discharge Disposition",
        "Discharge Diagnosis", "Discharge Condition"
    ]
    filtered = {}
    if 'start' in sec_dict:
        start_lines = sec_dict['start'].splitlines()
        for line in start_lines:
            if "Service:" in line:
                filtered['start'] = line.strip()
                break
    for key in keys_to_keep:
        if key in sec_dict:
            filtered[key] = sec_dict[key]
    return format_dict_output(filtered)

# Apply both functions to new columns
df['BHC_discharge'] = df['section_dict'].apply(build_bhc_discharge)
df['DI_discharge'] = df['section_dict'].apply(build_di_discharge)



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
filtered_df = df


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


# Function to format the input text
def format_input_bhc(row):
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
        f"## Discharge text\n: {truncate_by_bert_tokens(row.get('BHC_discharge', ''), 4000)}\n"
    )


# Function to format the input text
def format_input_di(row):
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
        f"## Discharge text\n: {truncate_by_bert_tokens(row.get('DI_discharge', ''), 4000)}\n"
    )









inst_bhc= """
You are a clinical language model. Below is the summarized discharge note and radiology note from the MIMIC-IV dataset. 
Generate the Brief Hospital Course section, focusing only on:
- Clinical events
- Interventions and procedures
- Patient progress during admission
Exclude discharge instructions and follow-up care.
"""


inst_ds= """
You are a clinical language model. Below is the summarized discharge note and radiology note from the MIMIC-IV dataset. 
Generate patient-facing Discharge instructions based on both notes.
Include reason for admission, clinical events, interventions, discharge condition, and follow-up care.
"""



# Generate BHC prompt examples
bhc_df = filtered_df.copy()
bhc_df["text"] = bhc_df.apply(
    lambda row: alpaca_prompt.format(
        instruction=inst_bhc.strip(),   #########################  inst_bhc for structured ##### inst_bhc1 for unstructured
        inputs=format_input_bhc(row),
        outputs=""   
    ),
    axis=1
)

# Generate DS prompt examples
di_df = filtered_df.copy()
di_df["text"] = di_df.apply(
    lambda row: alpaca_prompt.format(
        instruction=inst_ds.strip(),  #########################  inst_ds for structured ##### inst_ds1 for unstructured
        inputs=format_input_di(row),
        outputs=""
    ),
    axis=1
)







#### you can put this part in a function and call it over and over again, but i am little lazy, sorry

df = pd.DataFrame()
df["hadm_id"]=bhc_df["hadm_id"]
pred_bhc=[""]*len(bhc_df)
pred_ds=[""]*len(di_df)

for i in range(len(bhc_df)):
    txt1 = "Discharge text: \n" + bhc_df['text'][i]  
    out1 = summerize(txt1, inst_bhc)
    pred_bhc[i] = str(out1)
    txt1 = "Discharge text: \n" + di_df['text'][i]
    out2 = summerize(txt1, inst_ds)
    pred_ds[i] = str(out2)
    del txt1, out1, out2
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if i%100==0:
        df["brief_hospital_course"] = pred_bhc
        df["discharge_instructions"] = pred_ds
        df.to_excel("test2_unsloth_gemma2_joint_summary_on_truncated_1.xlsx", index=False)  #### output for instruction set 1

df["brief_hospital_course"] = pred_bhc
df["discharge_instructions"] = pred_ds
df.to_excel("test2_unsloth_gemma2_joint_summary_on_truncated_1.xlsx", index=False)  #### output for instruction set 1










end_part1 = time.perf_counter()
elapsed_part1 = end_part1 - elapsed_part1
print(f"[{format_seconds(elapsed_part1)}] prompt 1 time.")
elapsed_part1 = end_part1








inst_bhc= """
You are a clinical documentation assistant. Your task is to extract the Brief Hospital Course (BHC) from a hospital discharge note and radiology notes.
The Brief Hospital Course should provide a concise and structured summary of the patient's inpatient journey, including relevant history, major diagnoses, key interventions, clinical assessments, procedures performed, significant events during admission, and discharge status.
Please follow the format below for a structured output:

1. Chief Complaint / Presentation: [Brief statement of the reason for admission or presenting symptoms]
2. Relevant History: [Comorbidities or relevant medical/surgical history impacting care]
3. Hospital Course Summary:
  3.1. Day-by-day (or stage-by-stage) narrative of key clinical events, interventions, treatments, response to treatment, procedures (e.g., surgeries, ERCPs), and consultations.
  3.2. Summarize any imaging, labs, or radiology findings only if they impacted the clinical decision-making.
  3.3. Mention symptom resolution, response to therapy, and transition planning (e.g., mobility, PT, diet, discharge medications).
4. Discharge Status: [Patient condition at discharge and where they were discharged to, if available]
"""

inst_ds = """
You are a clinical documentation assistant. Your task is to extract and generate the Discharge Instructions section from a patient's discharge note and radiology notes.
The Discharge Instructions should be:
Written in simple, clear, and patient-facing language
Summarize key hospital events and explain what the patient should do after discharge
Include medication changes, follow-up appointments, self-care advice, and warning signs if applicable
Preserve clinical accuracy while ensuring understandability
Please follow the format below for a structured output:

1. Why you came to the hospital: [Summarize the reason for admission in one sentence]
2. What happened in the hospital: [Brief summary of key findings, diagnoses, procedures, and treatment]
3. Medications:
  3.1. Continue: [List important medications to continue]
  3.2. Start: [Newly prescribed medications]
  3.3. Stop or change: [Medications discontinued or dosage changes]
4. What to do after discharge:
  4.1. [Instructions on medications, diet, mobility, wound care, etc.]
  4.2. [Expected symptoms and when to seek help]
  4.3. [Additional care instructions, e.g., PICC line, oxygen use]
5. Follow-up appointments:
  5.1. [List scheduled follow-ups or instructions to make appointments, with providers and timeframes]
6. Other important information:
  6.1 [E.g., contact numbers, documentation sent, special precautions]
"""





# Generate BHC prompt examples
bhc_df = filtered_df.copy()
bhc_df["text"] = bhc_df.apply(
    lambda row: alpaca_prompt.format(
        instruction=inst_bhc.strip(),   #########################  inst_bhc for structured ##### inst_bhc1 for unstructured
        inputs=format_input_bhc(row),
        outputs=""   
    ),
    axis=1
)

# Generate DS prompt examples
di_df = filtered_df.copy()
di_df["text"] = di_df.apply(
    lambda row: alpaca_prompt.format(
        instruction=inst_ds.strip(),  #########################  inst_ds for structured ##### inst_ds1 for unstructured
        inputs=format_input_di(row),
        outputs=""
    ),
    axis=1
)




df = pd.DataFrame()
df["hadm_id"]=bhc_df["hadm_id"]
pred_bhc=[""]*len(bhc_df)
pred_ds=[""]*len(di_df)

for i in range(len(bhc_df)):
    txt1 = "Discharge text: \n" + bhc_df['text'][i]  
    out1 = summerize(txt1, inst_bhc)
    pred_bhc[i] = str(out1)
    txt1 = "Discharge text: \n" + di_df['text'][i]
    out2 = summerize(txt1, inst_ds)
    pred_ds[i] = str(out2)
    del txt1, out1, out2
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if i%100==0:
        df["brief_hospital_course"] = pred_bhc
        df["discharge_instructions"] = pred_ds
        df.to_excel("test2_unsloth_gemma2_joint_summary_on_truncated_2.xlsx", index=False)  #### output for instruction set 2

df["brief_hospital_course"] = pred_bhc
df["discharge_instructions"] = pred_ds
df.to_excel("test2_unsloth_gemma2_joint_summary_on_truncated_2.xlsx", index=False)  #### output for instruction set 2










end_part1 = time.perf_counter()
elapsed_part1 = end_part1 - elapsed_part1
print(f"[{format_seconds(elapsed_part1)}] prompt 2 time.")
elapsed_part1 = end_part1







inst_bhc = """
Generate the "Brief Hospital Course" section of a hospital discharge summary from the following Discharge note, using the following structure and clinical style:

1. Begin by stating the patient,s reason for admission using clinical language.
2. Summarize the major active medical problems addressed during the hospital stay.
3. For each problem:
   - Describe the diagnostic workup performed (labs, imaging, consults).
   - Outline the treatments or interventions (medications, procedures, surgeries).
   - Comment on the patient,s clinical response and recovery status.
4. Include any relevant chronic issues managed or monitored during the stay.
5. Conclude with the discharge disposition (e.g., home, rehab) and any brief follow-up plans.

Use concise, problem-oriented medical language appropriate for clinicians. Group content by condition using headings such as:
   - # [Medical Problem]
   - # Chronic Conditions
   - # Transition of Care

Do not include patient instructions or non-clinical explanations. Focus only on in-hospital course and medical decision-making.
"""


inst_ds = """
Generate the "Discharge Instructions" section for a patient discharge summary from the following Discharge note, using the following structure:

1. Greet the patient and state the reason for admission.
2. Briefly explain what was done during the hospital stay, including key procedures or findings.
3. Provide detailed follow-up instructions, including:
   - Medication changes (new, changed, or stopped)
   - Lifestyle advice (diet, physical activity, restrictions)
   - Symptom monitoring (what to watch for, when to seek help)
   - Follow-up appointments and contact information
4. Use plain language that the patient can understand.
5. Maintain a compassionate, supportive tone.
6. (Optional) Use headings like:
   - Why did you come to the hospital?
   - What happened here?
   - What should you do now?
7. End with a warm closing message from the care team.

The generated instructions should be written in a clear and empathetic style appropriate for a patient leaving the hospital.
"""





# Generate BHC prompt examples
bhc_df = filtered_df.copy()
bhc_df["text"] = bhc_df.apply(
    lambda row: alpaca_prompt.format(
        instruction=inst_bhc.strip(),   #########################  inst_bhc for structured ##### inst_bhc1 for unstructured
        inputs=format_input_bhc(row),
        outputs=""   
    ),
    axis=1
)

# Generate DS prompt examples
di_df = filtered_df.copy()
di_df["text"] = di_df.apply(
    lambda row: alpaca_prompt.format(
        instruction=inst_ds.strip(),  #########################  inst_ds for structured ##### inst_ds1 for unstructured
        inputs=format_input_di(row),
        outputs=""
    ),
    axis=1
)





df = pd.DataFrame()
df["hadm_id"]=bhc_df["hadm_id"]
pred_bhc=[""]*len(bhc_df)
pred_ds=[""]*len(di_df)

for i in range(len(bhc_df)):
    txt1 = "Discharge text: \n" + bhc_df['text'][i]  
    #txt2 = "Discharge text: \n" + truncated_to_limit(del_trash(df["real_discharge_text"].astype(str)), 3500) + "\n" + "Radiology text: \n" + truncated_to_limit(del_trash(df["real_radiology_text"].astype(str)),2500)
    out1 = summerize(txt1, inst_bhc)
    pred_bhc[i] = str(out1)
    #del txt1, out1
    txt1 = "Discharge text: \n" + di_df['text'][i]
    out2 = summerize(txt1, inst_ds)
    pred_ds[i] = str(out2)
    del txt1, out1, out2
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if i%100==0:
        df["brief_hospital_course"] = pred_bhc
        df["discharge_instructions"] = pred_ds
        df.to_excel("test2_unsloth_gemma2_joint_summary_on_truncated_3.xlsx", index=False) #### output for instruction set 3

df["brief_hospital_course"] = pred_bhc
df["discharge_instructions"] = pred_ds
df.to_excel("test2_unsloth_gemma2_joint_summary_on_truncated_3.xlsx", index=False) #### output for instruction set 3











end_part1 = time.perf_counter()
elapsed_part1 = end_part1 - elapsed_part1
print(f"[{format_seconds(elapsed_part1)}] prompt 3 time.")
elapsed_part1 = end_part1









inst_bhc = """
You are a clinical documentation specialist. Based on the discharge summary below, generate a concise and accurate "Brief Hospital Course" section.

The section should include:
- Reason for admission and presenting symptoms
- Key diagnostic workup and findings
- Summary of treatment provided during admission
- Patient's response to treatment
- Discharge condition and disposition

Write clearly in clinical style suitable for inclusion in a discharge summary. Use paragraph format or structured bullets.

Here is the Discharge Report:
"""


inst_ds = """
You are a clinical documentation assistant.

Read the full discharge summary provided below and generate a clear, patient-facing "Discharge Instructions" section.

Instructions should:
- Begin with a statement of why the patient was admitted.
- Mention whether the cause was identified, and summarize relevant findings or evaluations.
- Include any major tests, procedures, or consultations that occurred or are planned.
- Recommend follow-up care and outpatient appointments.
- List the medications prescribed at discharge for symptom management.
- Use plain language suitable for patients or caregivers.
- Keep the tone supportive and informative.

Here is the discharge report:
"""






# Generate BHC prompt examples
bhc_df = filtered_df.copy()
bhc_df["text"] = bhc_df.apply(
    lambda row: alpaca_prompt.format(
        instruction=inst_bhc.strip(),   #########################  inst_bhc for structured ##### inst_bhc1 for unstructured
        inputs=format_input_bhc(row),
        outputs=""   
    ),
    axis=1
)

# Generate DS prompt examples
di_df = filtered_df.copy()
di_df["text"] = di_df.apply(
    lambda row: alpaca_prompt.format(
        instruction=inst_ds.strip(),  #########################  inst_ds for structured ##### inst_ds1 for unstructured
        inputs=format_input_di(row),
        outputs=""
    ),
    axis=1
)




df = pd.DataFrame()
df["hadm_id"]=bhc_df["hadm_id"]
pred_bhc=[""]*len(bhc_df)
pred_ds=[""]*len(di_df)

for i in range(len(bhc_df)):
    txt1 = "Discharge text: \n" + bhc_df['text'][i]  
    #txt2 = "Discharge text: \n" + truncated_to_limit(del_trash(df["real_discharge_text"].astype(str)), 3500) + "\n" + "Radiology text: \n" + truncated_to_limit(del_trash(df["real_radiology_text"].astype(str)),2500)
    out1 = summerize(txt1, inst_bhc)
    pred_bhc[i] = str(out1)
    #del txt1, out1
    txt1 = "Discharge text: \n" + di_df['text'][i]
    out2 = summerize(txt1, inst_ds)
    pred_ds[i] = str(out2)
    del txt1, out1, out2
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if i%100==0:
        df["brief_hospital_course"] = pred_bhc
        df["discharge_instructions"] = pred_ds
        df.to_excel("test2_unsloth_gemma2_joint_summary_on_truncated_4.xlsx", index=False)#### output for instruction set 4

df["brief_hospital_course"] = pred_bhc
df["discharge_instructions"] = pred_ds
df.to_excel("test2_unsloth_gemma2_joint_summary_on_truncated_4.xlsx", index=False)  #### output for instruction set 4





end_part1 = time.perf_counter()
elapsed_part1 = end_part1 - elapsed_part1
print(f"[{format_seconds(elapsed_part1)}] prompt 4 time.")
elapsed_part1 = end_part1


