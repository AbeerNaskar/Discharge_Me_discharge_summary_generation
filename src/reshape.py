##### further processing, the processed documents using fine_tun_out_to_scoring_inp.py
##### if any error comes with the previously generated file


import os
import pandas as pd

def clean_text(out: str) -> str:
    if not isinstance(out, str):
        return out
    
    # Extract after ### Response:
    if '### Instruction:' in out and '### Response:' in out:
        start = out.index("### Response:") + len("### Response:")
        out = out[start:].strip()

    # Remove unwanted tokens
    tokens = ["<|im_end|>", "<|begin_of_text|>", "<|end_of_text|>",
              "<end_of_turn>", "<bos>", "<eos>", "<|eot_id|>", "<s>", "</s>"]
    
    for token in tokens:
        out = out.replace(token, "")
    
    return out.strip()


# Process all Excel files in current directory, (it modify the current files, all the .xlsx files in the current directory)
for file in os.listdir("./"):
    if file.endswith(".xlsx"):
        path = os.path.join("./", file)
        print(f"Processing: {path}")

        df = pd.read_excel(path)
        df = df.fillna("No generation")

        # Only clean if columns exist
        for col in ["brief_hospital_course", "discharge_instructions"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        # Save back to the same file
        df.to_excel(path, index=False)
        print(f"Updated and saved: {path}")

print("Done!")
