# Discharge_Me!_discharge_summary_generation

Generate Discharge summaries, (brief_hospital_course and discharge_instructions) from discharge report

## Codes

All the codes are in ```src``` folder.

## Code description

You need to download the original data from the [Data link](https://physionet.org/content/discharge-me/1.3/)
<br/>
It should have train, valid, test_phase_2 folder
<br/>
ownload unsloth models also (or you can use the model id and provide your huggingface api, it will be downloaded automatically, but I like to go with offline model)
<br/>
keep those data or models in same folder and change the path variable accordingly in those codes if needed.

### Prepare data for train, valid and test for fine tuning

run ```create_test_data_full.py``` for creating the fine tune input from the downloaded data (full input strategy)
<br/>

run ```create_test_data_truncated.py``` for creating the fine tune input from the downloaded data (truncated input strategy)
<br/>

run ```fine-tune-full-prompt.py``` for fine tuning on full input (full input strategy)

<br/>

run ```fine-tune-truncated-prompt.py``` for fine tuning on truncated input (truncated input strategy)
<br/>

run ```zero-shot-full-prompt.py``` for zero shot on full input (full input strategy)
<br/>

run ```zero-shot-truncated-prompt.py``` for zero shot on truncated input (truncated input strategy)
<br/>

run ```out_to_score_inp.py``` for converting the generated output to scoring code input
you can run ```reshape.py``` for convertion (double check)
<br/>

<br/>
on this data
<br/>

run ```summac_calculate.py``` for each output, to get section wise SummaC scores
<br/>

run ```scoring_calculate_segmentwise_multiprocessing.py``` corresponding to entire output folder to get section wise scores of each files
<br/>

run ```cus_draw.py``` corresponding to entire output folder to get section wise uniqueness score of each files together in a plot
