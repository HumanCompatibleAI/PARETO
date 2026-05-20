---
pretty_name: Politically Neutral AI
language:
- en
tags:
- image
- tabular
- text
- political-science
- human-preference
- ai-safety
size_categories:
- 1K<n<10K
source_datasets:
- original
configs:
- config_name: responses_anthropic
  data_files:
  - split: train
    path: responses/anthropic.csv
- config_name: responses_gemini
  data_files:
  - split: train
    path: responses/gemini.csv
- config_name: responses_grok
  data_files:
  - split: train
    path: responses/grok.csv
- config_name: responses_llama
  data_files:
  - split: train
    path: responses/llama.csv
- config_name: responses_openai_balanced
  data_files:
  - split: train
    path: responses/openai_balanced.csv
- config_name: responses_openai_default
  data_files:
  - split: train
    path: responses/openai_default.csv
- config_name: responses_openai_oppose
  data_files:
  - split: train
    path: responses/openai_oppose.csv
- config_name: responses_openai_support
  data_files:
  - split: train
    path: responses/openai_support.csv
- config_name: study_stance_dict
  data_files:
  - split: train
    path: study_results/stance_dict.csv
- config_name: study_user_questions_provenance
  data_files:
  - split: train
    path: study_results/user_questions_provenance.csv
- config_name: stimuli
  drop_labels: true
  data_files:
  - split: train
    path: stimuli/**/*.png
---

# NeurIPS Submission Data

This folder was assembled from the `chang_neutral` workspace.

## Contents

- `responses/`: raw model response CSVs. Each file has 200 rows and includes `regeneration_count`.
- `rendered_answers/`: rendered answer JSONs keyed by question id.
- `study_results/`: study metadata and derived analysis tables.
- `stimuli/`: 1600 PNG stimuli copied into eight 200-image model folders.

## Included Study Results

- `user_questions_provenance.csv`: question id, displayed question text, source, custom link, and original source question.
- `stance_dict.csv`: stance response text, polarity, and model stance labels.
- `qualtrics_anonymized.csv`: anonymized Qualtrics study results.
- `prolific_anonymized.csv`: anonymized Prolific study results.



