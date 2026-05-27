# PARETO

This is the PARETO data set which is described in the paper *Political Neutrality as Balanced Approval:
A Large-Scale Human Evaluation of AI Responses*

## Contents

- `responses/`: raw model/stance response CSVs. Each file has 200 rows and includes `regeneration_count`.
- `rendered_answers/`: question text combined with model responses, keyed by question ID
- `stimuli/`: 1600 PNG stimuli in eight 200-image model folders
- `study_results/`: participant answers and demographics

## Included Study Results

- `user_questions_provenance.csv`: question id, displayed question text, source, custom link, and original source question.
- `stance_dict.csv`: stance response text, polarity, and model stance labels.
- `likert_questions.csv`: text of survey questions asked
- `likert-responses.csv`: what each participant answered for the four AI responses shown, keyed by question and participant ID
- `free_text_responses.csv`: participant qualitative feedback on each response, keyed by question and participant ID
- `participant_demographics.csv`: keyed on participant ID

Note: To preserve privacy, each Prolific ID has been hashed into a new unique participant ID that is consistent across files.
