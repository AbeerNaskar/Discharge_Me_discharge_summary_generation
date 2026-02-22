#### code to clean the output , split it into two columns form one column output.


import pandas as pd
import os
from glob import glob

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove special tokens
    for token in ["<|im_end|>", "<|begin_of_text|>", "<|end_of_text|>", "<end_of_turn>", "<bos>", "<eos>", "<|eot_id|>"]:
        text = text.replace(token, "")

    # Deduplicate lines
    lines = text.strip().split('\n')
    seen = set()
    cleaned_lines = []
    for line in lines:
        normalized = line.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

# Process all .xlsx files in the current directory
for file_path in glob("./*.xlsx"): ## it read all files in current directory, of course you can change the directory 
    #if file_path.startswith("score_"): # if you want to skip any file start with some suffix
    #    continue  # Skip already processed files

    try:
        df = pd.read_excel(file_path)
        if 'prediction' not in df.columns or 'hadm_id' not in df.columns:
            print(f"Skipping {file_path}: missing required columns.")
            continue

        df['cleaned_prediction'] = df['prediction'].apply(clean_text)

        # Construct new columns
        hadm_id_list = list(df['hadm_id'][250:])   # hard coded as there are 250 documents in test set
        brief_hospital_course_list = list(df['cleaned_prediction'][:250])
        discharge_instructions_list = list(df['cleaned_prediction'][250:])

        # Build new DataFrame
        result_df = pd.DataFrame({
            'hadm_id': hadm_id_list,
            'brief_hospital_course': brief_hospital_course_list,
            'discharge_instructions': discharge_instructions_list
        })

        # Save result
        output_filename = f"new_ft_dec_{os.path.basename(file_path)}"  #### if you want to change name suffix, change here
        result_df.to_excel(output_filename, index=False)
        print(f"Saved: {output_filename}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
