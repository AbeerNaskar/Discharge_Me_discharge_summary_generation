import pandas as pd
from summac.model_summac import SummaCZS, SummaCConv

PATH1 = 'path_to_generated_output'
PATH = 'Path_to_test_file_which_contains_discharge_text'

# Load files
csv_file = PATH + "test.csv"   # test file with 'discharge_text'
xlsx_file = PATH1 + "extra/new_ft_dec_test2_unsloth_phi4_as_it_is1_.xlsx"  # Generated xlsx file


df_csv = pd.read_csv(csv_file)
df_xlsx = pd.read_excel(xlsx_file)

# Keep only relevant columns
df_csv = df_csv[['hadm_id', 'discharge_text']]

# Split df_csv into BHC and DI
df_bhc = df_csv.iloc[:250].copy()  # hard coded: as there are 250 samples, you can also use len(df_csv) instead of 250
df_di = df_csv.iloc[-250:].copy()  # ard coded: as there are 250 samples, you can also use len(df_csv) instead of 250

# Prepare model summaries
df_bhc1 = df_xlsx[['hadm_id', 'brief_hospital_course']].copy()
df_bhc1.rename(columns={'brief_hospital_course': 'summary'}, inplace=True)

df_di1 = df_xlsx[['hadm_id', 'discharge_instructions']].copy()
df_di1.rename(columns={'discharge_instructions': 'summary'}, inplace=True)

# Merge on hadm_id
df_bhc_m = pd.merge(df_bhc, df_bhc1, on='hadm_id')
df_di_m = pd.merge(df_di, df_di1, on='hadm_id')

# Combine
df_merge = pd.concat([df_bhc_m, df_di_m], ignore_index=True)
df_merge.dropna(subset=['hadm_id', 'discharge_text', 'summary'], inplace=True)
df_merge = df_merge.sample(frac=1, random_state=42).reset_index(drop=True)#[:250]

# Load SUMMAC models (if you have gpu then comment out next two lines, and remove comment from third and fourth line)
# if you do not have gpu
model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu")
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")
# if you have gpu
#model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda")
#model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

# Containers for segment scores
seg1_scores_zs, seg1_scores_conv = [], []
seg2_scores_zs, seg2_scores_conv = [], []
seg3_scores_zs, seg3_scores_conv = [], []
seg4_scores_zs, seg4_scores_conv = [], []

t=0
# Process documents
for _, row in df_merge.iterrows():
    doc = row['discharge_text']
    summary = row['summary']
    print(t)
    t+=1

    # Define segments

    # if you want 0-1k, 1k-2k, 2k-3k, 3k-last split
    #seg1 = summary[0:1000]
    #seg2 = summary[1000:2000]
    #seg3 = summary[2000:3000]
    #seg4 = summary[3000:]

    # if you want 0-1k, 0-2k, 0-3k, entire segment
    seg1 = summary[:1000]
    seg2 = summary[:2000]
    seg3 = summary[:3000]
    seg4 = summary

    # Score segment 1
    if seg1.strip():
        s_zs = model_zs.score([doc], [seg1])["scores"][0]
        s_conv = model_conv.score([doc], [seg1])["scores"][0]
        seg1_scores_zs.append(s_zs)
        seg1_scores_conv.append(s_conv)

    # Score segment 2
    if seg2.strip():
        s_zs = model_zs.score([doc], [seg2])["scores"][0]
        s_conv = model_conv.score([doc], [seg2])["scores"][0]
        seg2_scores_zs.append(s_zs)
        seg2_scores_conv.append(s_conv)

    # Score segment 3
    if seg3.strip():
        s_zs = model_zs.score([doc], [seg3])["scores"][0]
        s_conv = model_conv.score([doc], [seg3])["scores"][0]
        seg3_scores_zs.append(s_zs)
        seg3_scores_conv.append(s_conv)

    # Score segment 3
    if seg4.strip():
        s_zs = model_zs.score([doc], [seg4])["scores"][0]
        s_conv = model_conv.score([doc], [seg4])["scores"][0]
        seg4_scores_zs.append(s_zs)
        seg4_scores_conv.append(s_conv)

# Calculate averages per segment
avg_seg1_zs = sum(seg1_scores_zs) / len(seg1_scores_zs) if seg1_scores_zs else None
avg_seg1_conv = sum(seg1_scores_conv) / len(seg1_scores_conv) if seg1_scores_conv else None

avg_seg2_zs = sum(seg2_scores_zs) / len(seg2_scores_zs) if seg2_scores_zs else None
avg_seg2_conv = sum(seg2_scores_conv) / len(seg2_scores_conv) if seg2_scores_conv else None

avg_seg3_zs = sum(seg3_scores_zs) / len(seg3_scores_zs) if seg3_scores_zs else None
avg_seg3_conv = sum(seg3_scores_conv) / len(seg3_scores_conv) if seg3_scores_conv else None

avg_seg4_zs = sum(seg4_scores_zs) / len(seg4_scores_zs) if seg4_scores_zs else None
avg_seg4_conv = sum(seg4_scores_conv) / len(seg4_scores_conv) if seg4_scores_conv else None


# Print results
print("=== Segment-wise Average Scores fron beginning ===", xlsx_file)
print(f"Segment 1: SummaCZS = {avg_seg1_zs:.3f}, SummaCConv = {avg_seg1_conv:.3f}")
print(f"Segment 2: SummaCZS = {avg_seg2_zs:.3f}, SummaCConv = {avg_seg2_conv:.3f}")
print(f"Segment 3: SummaCZS = {avg_seg3_zs:.3f}, SummaCConv = {avg_seg3_conv:.3f}")
print(f"Segment 4: SummaCZS = {avg_seg4_zs:.3f}, SummaCConv = {avg_seg4_conv:.3f}")



















