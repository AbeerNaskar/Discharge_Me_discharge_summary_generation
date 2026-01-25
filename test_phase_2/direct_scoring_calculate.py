import pandas as pd

# Step 1: Load all input files
radiology = pd.read_csv('radiology.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
discharge = pd.read_csv('discharge.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
target = pd.read_csv('discharge_target.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')

# Step 2: Rename discharge 'text' column
discharge = discharge[['hadm_id', 'text']].rename(columns={'text': 'discharge_text'})

# Step 3: Group radiology by 'hadm_id' and concatenate texts
radiology_grouped = (
    radiology
    .groupby('hadm_id')['text']
    .apply(lambda x: ' '.join(str(t) for t in x))
    .reset_index()
    .rename(columns={'text': 'radiology_text'})
)

# Step 4: Select relevant columns from target
target = target[['hadm_id', 'discharge_instructions', 'brief_hospital_course']]

# Step 5: Merge all three datasets on 'hadm_id'
df = target.merge(discharge, on='hadm_id', how='inner')
df = df.merge(radiology_grouped, on='hadm_id', how='inner')

# Step 6: Read list of hadm_id from test3.txt
with open('test3.txt', 'r') as f:
    test_ids = eval(f.read())

# Step 7: Filter rows based on test3.txt hadm_id list
df = df[df['hadm_id'].isin(test_ids)].reset_index(drop=True)

# Step 8: Create target.csv where both columns are the same (discharge_text + radiology_text)
concat_text = df['discharge_text'].fillna('') + ' ' + df['radiology_text'].fillna('')
target_df = df[['hadm_id']].copy()
target_df['discharge_instructions'] = concat_text
target_df['brief_hospital_course'] = concat_text
target_df.to_csv('target_direct_ckecking.csv', index=False)

# Step 9: Create Excel file as requested
df[['hadm_id', 'discharge_instructions', 'brief_hospital_course']].to_excel('submision_direct_ckecking.xlsx', index=False)

print("Files 'target.csv' and 'submision.xlsx' created successfully.")
