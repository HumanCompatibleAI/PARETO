# PARETO

This is the PARETO data set which is described in the paper *Political Neutrality as Balanced Approval: A Large-Scale Human Evaluation of AI Responses*

## Contents
- `analysis/`: code to reproduce the figures in the paper
  - `final_analyses.ipynb`: create (almost) all figures, write to `figures` dir
  - `analysis.py`: main analysis code 
  - `demographic_analysis.ipynb`: create demographics figures
  - `demographic_analysis.py`: main demographics code
  - `issue_alignment.ipynb`: create issue alignment PCA and correlation figures
  - `stance_dict.csv`: stance response text, polarity, and model stance labels.
- `model_responses/`: raw model/stance response CSVs. Each file has 200 rows.
- `stimuli_json/`: question text combined with model responses, keyed by question ID
- `stimuli_png/`: 1600 PNG stimuli in eight 200-image model folders
- `survey_data/`: participant answers, demographics, and analysis keys
  - `user_questions_provenance.csv`: question id, displayed question text, source, custom link, and original source question.
  - `likert_questions.csv`: text of survey questions asked
  - `likert-responses.csv`: what each participant answered for the four AI responses shown, keyed by question and participant ID
  - `free_text_responses.csv`: participant qualitative feedback on each response, keyed by question and participant ID
  - `participant_demographics.csv`: keyed on participant ID

Note: To preserve privacy, each Prolific ID has been hashed into a new unique participant ID that is consistent across files.
