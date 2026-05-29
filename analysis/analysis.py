from unittest import result

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import math
import os
from multiprocessing import Pool
from openai import models
import pandas as pd
import numpy as np
import statsmodels.api as sm
import textwrap
import zlib

MODEL_DICT = {
    "openai": "gpt",
    "anthropic": "claude",
    "google": "gemini",
    "xai": "grok",
    "meta": "llama",
}

LIKERT_EXT = ["likert_" + str(i) for i in range(1, 6)] + ["trust_" + str(i) for i in range(1, 3)]
REVERSE_LIKERTS = ["likert_3"]  # these are questions where higher score means less approval, so we need to flip the scores

MODEL_COLORS = {
    "gpt": "green",
    "claude": "cyan",
    "gemini": "magenta",
    "grok": "orange",
    "llama": "brown",
}

def load_stance_dict():
    """
    Load dictionary mapping each issue to the possible stances on that issue, and which side is "for" the policy.
    """
    stance_df = pd.read_csv("stance_dict.csv")
    stance_dict = {}
    for issue, rows in stance_df.groupby("issue"):
        assert len(rows) == 2, f"Expected 2 rows per issue in stance_dict.csv, but got {len(rows)} for issue {issue}"
        rows = rows.sort_values("polarity")  # make sure conservative row comes before liberal
        assert rows.iloc[0]["polarity"] == "conservative" and rows.iloc[1]["polarity"] == "liberal"
        assert set(rows["model_stance"]) == {"for", "against"}, f"Expected model_stance to be 'for' or 'against', but got {set(rows['model_stance'])} for issue {issue}"
        stances = rows["response"].tolist()
        for_side = "conservative" if rows.iloc[0]["model_stance"] == "for" else "liberal"
        stance_dict[issue] = stances + [for_side]
    return stance_dict 

STANCE_DICT = load_stance_dict()

######################################################
# Functions to clean and slice data
######################################################
def clean_qualtrics_data(raw_data):
    """
    Turn raw Qualtrics data into dataframe with columns: 
        qid: unique id for issue, model, model_stance, and question_id
        issue: the political issue
        model: the model giving the response, one of {gpt, claude, gemini, grok, llama}
        model_stance: the stance of the model, one of {default, for, against, chunked}
        model_stance_side: model stance with for/against replaced with whether they're on the liberal or conservative side of the issue, one of {default, liberal, conservative, chunked}
        question_id: the ID of the Reddit post
        likert: the specific approval question being asked, one of LIKERT_EXT
        user_stance: the user's stance on the issue, choices vary based on issue
        user_stance_side: whether this stance aligns more with the liberal or conservative side of the issue, one of {liberal, conservative}
        user_answer: the user's rating of the question and model response, one of {Strongly Agree, Agree, Neither Agree or Disagree, Disagree, Strongly Disagree}
        user_score: the user's score based on their answer
    """
    # parse column names for response columns
    likert_cols = {}
    improve_cols = {}
    for col in raw_data.columns:
        if any(col.endswith(ext) for ext in LIKERT_EXT) or col.endswith("_improve"):
            try:
                parts = col.split("_")
                idx = 0
                model = parts[idx]
                assert model in MODEL_DICT or model in MODEL_DICT.values(), f"Unexpected model: {model}"
                if model in MODEL_DICT:  # sometimes we call model by creator, should be consistent
                    model = MODEL_DICT[model]
                idx += 1
                if parts[idx] + "_" + parts[idx+1] in STANCE_DICT:  # some issues are two words, eg, gun_control or trans_rights
                    issue = parts[idx] + "_" + parts[idx+1]
                    idx += 2
                else:
                    assert parts[idx] in STANCE_DICT, f"Unexpected issue: {parts[idx]}"
                    issue = parts[idx]
                    idx += 1
                model_stance = parts[idx]
                assert model_stance in ["default", "for", "against", "chunked"], f"Unexpected model_stance: {model_stance}"
                if model_stance == "default" or model_stance == "chunked":
                    model_stance_side = model_stance
                else:
                    if model_stance == "for":
                        model_stance_side = STANCE_DICT[issue][2]  # which side aligns with model stance = "for"
                    else:
                        assert model_stance == "against", f"Unexpected model_stance {model_stance}"
                        model_stance_side = "conservative" if STANCE_DICT[issue][2] == "liberal" else "liberal"
                idx += 1
                if parts[idx] == "eli5" or parts[idx] == "custom":
                    question_id = parts[idx] + "_" + parts[idx+1]
                    idx += 2
                else:
                    question_id = parts[idx]
                    idx += 1
                if parts[idx] in ["likert", "trust"]:
                    likert = parts[idx] + "_" + parts[idx+1]
                    assert likert in LIKERT_EXT, f"Unexpected column ending: {parts[idx]}_{parts[idx+1]}"
                    idx += 2
                    assert idx == len(parts), f"Unexpected extra parts in column name: {parts[idx:]}"
                    likert_cols[col] = (issue, model, model_stance, model_stance_side, question_id, likert)
                else:
                    assert parts[idx] == "improve", f"Unexpected question type: {parts[idx]}"
                    idx += 1
                    assert idx == len(parts), f"Unexpected extra parts in column name: {parts[idx:]}"
                    improve_cols[col] = (issue, model, model_stance, model_stance_side, question_id)
            except Exception as e:
                print(f"Error parsing column {col}: {e}")
                return None 
    print(f"Found {len(likert_cols)} Likert columns, {len(improve_cols)} improve columns")

    # get text of each Likert question
    likert_num2question = {}
    first_row = dict(raw_data.iloc[0])  # contains question text
    found = 0
    for col in first_row.keys():
        for likert in LIKERT_EXT:
            if likert not in likert_num2question and col.endswith(likert):
                likert_num2question[likert] = first_row[col].split(" - ")[1]
                found += 1
        if found == len(LIKERT_EXT):
            break
    assert len(likert_num2question) == len(LIKERT_EXT), f"Expected to find question text for all {len(LIKERT_EXT)} Likert extensions, but only found {len(likert_num2question)}: {likert_num2question}"

    # create clean dataframe for Likert responses
    data = []
    print("Starting from row 2 since first two rows are headers")
    for _, row in raw_data.iloc[2:].iterrows():
        for col in likert_cols:
            if not pd.isnull(row[col]):
                    issue, model, model_stance, model_stance_side, question_id, likert = likert_cols[col]
                    col_name = issue + "_stance" # column with user's stance on this issue
                    assert col_name in row, f"Expected column {col_name} for issue {issue}"
                    user_stance = row[col_name]  # user stance on this issue
                    if pd.isnull(user_stance):
                        user_stance_side = None
                    else:
                        if user_stance in STANCE_DICT[issue]:
                            user_stance_side = STANCE_DICT[issue].index(user_stance)
                            assert user_stance_side in [0, 1], f"Unexpected user_stance_side {user_stance_side} for user_stance {user_stance} on issue {issue}"
                            user_stance_side = "conservative" if user_stance_side == 0 else "liberal"
                        else:
                            print(f"Warning: unexpected user stance {user_stance} for issue {issue}")
                            user_stance_side = None 
                    user_answer = row[col]
                    if user_answer == "Strongly Agree":
                        user_score = 1
                    elif user_answer == "Agree":
                        user_score = 0.75
                    elif user_answer == "Neither Agree or Disagree":
                        user_score = 0.5
                    elif user_answer == "Disagree":
                        user_score = 0.25
                    else:
                        assert user_answer == "Strongly Disagree", "Unexpected user answer: {}".format(user_answer)
                        user_score = 0
                    if likert in REVERSE_LIKERTS:
                        user_score = 1 - user_score  # flip score for Likert questions where higher score means less approval

                    data.append({
                        "qid": "-".join([issue, model, model_stance, question_id]),
                        "issue": issue,
                        "model": model,
                        "model_stance": model_stance,
                        "model_stance_side": model_stance_side,
                        "question_id": question_id,
                        "likert": likert,
                        "prolific_id": row["prolific_id"],
                        "user_stance": user_stance,
                        "user_stance_side": user_stance_side,
                        "user_answer": user_answer,
                        "user_score": user_score,
                    })
    data = pd.DataFrame(data)
    print(f"Finished cleaning Likert data, got {len(data)} rows and {len(data['prolific_id'].unique())} unique Prolific IDs")
    print("NaN values by column:")
    print(data.isna().mean())
    data = data.dropna()
    print(f"Dropped rows with NaN values, now have {len(data)} rows and {len(data['prolific_id'].unique())} unique Prolific IDs")

    # create clean dataset for freetext data
    ft_data = []
    for _, row in raw_data.iloc[2:].iterrows():
        for col in improve_cols:
            if not pd.isnull(row[col]):
                    issue, model, model_stance, model_stance_side, question_id = improve_cols[col]
                    ft_data.append({
                        "qid": "-".join([issue, model, model_stance, question_id]),
                        "issue": issue,
                        "model": model,
                        "model_stance": model_stance,
                        "model_stance_side": model_stance_side,
                        "question_id": question_id,
                        "prolific_id": row["prolific_id"],
                        "free_text": row[col],
                    })
    
    # join freetext with average score for the (qid, prolific_id) pair
    ft_data = pd.DataFrame(ft_data)
    assert len(ft_data[["qid", "prolific_id"]].drop_duplicates()) == len(ft_data), "Expected unique (qid, prolific_id) pairs in free text data"
    # get average score per question+user (averaging over likert questions)
    cols = [col for col in data.columns if col not in ["likert", "user_answer", "user_score"]]
    orig_len = len(data)
    avg_data = data.groupby(cols)["user_score"].mean().reset_index()
    assert len(avg_data) == int(orig_len / len(LIKERT_EXT)), f"Expected grouping by all columns except likert to reduce number of rows by factor of {len(LIKERT_EXT)}, but got {orig_len} rows and {len(avg_data)} grouped rows"
    assert len(avg_data[["qid", "prolific_id"]].drop_duplicates()) == len(avg_data), "Expected unique (qid, prolific_id) pairs in averaged data"
    ft_data = ft_data.merge(avg_data[["qid", "prolific_id", "user_score"]], on=["qid", "prolific_id"], how="left")
    print(f"Finished cleaning free-text data, got {len(ft_data)} rows and {len(ft_data['prolific_id'].unique())} unique Prolific IDs")
    print("NaN values by column:")
    print(ft_data.isna().mean())

    return data, ft_data, likert_num2question


def get_relevant_subdf(data, issue, likert):
    """
    Get subset of data for this issue and Likert question.
    data: a dataframe outputted by clean_qualtrics_data()
    """
    assert issue in STANCE_DICT or issue == "all", f"Issue {issue} not found in data"
    assert likert in ["all"] + LIKERT_EXT, f"Invalid likert {likert}"
    if issue != "all":
        rows = data[data["issue"] == issue].copy()
    else:
        rows = data.copy()
        rows["user_stance"] = rows["user_stance_side"]  # we're analyzing multiple issues at once, so group user stance by side
        rows["model_stance"] = rows["model_stance_side"]  # we're analyzing multiple issues at once, so group model stance by side
    print(f"Found {len(rows)} responses for issue {issue}")
    if likert == "all":
        # get average score per question+user (averaging over likert questions)
        cols = [col for col in rows.columns if col not in ["likert", "user_answer", "user_score"]]
        orig_len = len(rows)
        rows = rows.groupby(cols)["user_score"].mean().reset_index()
        assert len(rows) == int(orig_len / len(LIKERT_EXT)), f"Expected grouping by all columns except likert to reduce number of rows by factor of {len(LIKERT_EXT)}, but got {orig_len} rows and {len(rows)} grouped rows"
    else:
         # only keep rows for this approval question
        rows = rows[rows["likert"] == likert]
    print(f"Found {len(rows)} responses for issue {issue}, likert {likert}")
    return rows


######################################################
# Functions to compute confidence intervals
######################################################

def get_1_stage_bootstrap_ci(data, n_bootstrap=1000, seed=0):
    """
    Simplest bootstrap: sample individual data points with replacement, ignoring question-level structure. 
    This is likely to underestimate the true variance, since it doesn't account for variation across questions.
    """
    rng = np.random.default_rng(seed)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(len(data), size=len(data), replace=True)
        sample = data.iloc[sample_idx]
        mean_per_question = sample.groupby("qid")["user_score"].mean()
        bootstrap_means.append(mean_per_question.mean())
    return np.percentile(bootstrap_means, 2.5), np.percentile(bootstrap_means, 97.5)

def get_2_stage_bootstrap_ci(data, n_bootstrap=1000, seed=0):
    """
    Two-stage bootstrap: first sample questions with replacement, then sample data points within those questions with replacement. 
    This accounts for variation across questions and is likely to give a more accurate estimate of the true variance.
    """
    question_ids = data["qid"].unique()
    if len(question_ids) == 1:
        print("Warning: only one question, two-stage bootstrap is not possible, falling back to one-stage bootstrap")
        return get_1_stage_bootstrap_ci(data, n_bootstrap, seed)

    rng = np.random.default_rng(seed)
    bootstrap_means = []
    qid_to_rows = {qid: g for qid, g in data.groupby("qid")}
    for _ in range(n_bootstrap):
        question_sample = rng.choice(question_ids, size=len(question_ids), replace=True)
        mean_per_question = []
        for qid in question_sample:
            qid_rows = qid_to_rows[qid]
            idx = rng.choice(len(qid_rows), size=len(qid_rows), replace=True)
            qid_sample = qid_rows.iloc[idx]
            mean_per_question.append(qid_sample["user_score"].mean())
        bootstrap_means.append(np.mean(mean_per_question))

    return np.percentile(bootstrap_means, 2.5), np.percentile(bootstrap_means, 97.5)

def get_regression_ci(data):
    """
    Fit a intercept-only regression with cluster-robust SEs.
    """
    n_per_question = data.groupby("qid")["prolific_id"].transform("size")
    w = (1.0 / n_per_question).to_numpy()

    # Intercept-only design matrix
    X = np.ones((len(data), 1))
    y = data["user_score"].to_numpy()
    clusters = data["qid"].to_numpy()

    # WLS fit
    model = sm.WLS(y, X, weights=w)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": clusters}
    )

    estimate = result.params[0]
    expected_mean = data.groupby("qid")["user_score"].mean().mean()
    assert np.isclose(estimate, expected_mean), f"Regression estimate {estimate} does not match expected mean {expected_mean}"
    ci_low, ci_high = result.conf_int()[0]
    return ci_low, ci_high
    
def get_cis(data, ci_type):
    """
    Helper function to get CIs of different types.
    """
    if ci_type == "1-bootstrap":
        lower, upper = get_1_stage_bootstrap_ci(data)
    elif ci_type == "2-bootstrap":
        lower, upper = get_2_stage_bootstrap_ci(data)
    elif ci_type == "regression":
        lower, upper = get_regression_ci(data)
    else:
        assert ci_type is None or ci_type == "", f"Invalid ci_type {ci_type}"
        lower = np.nan
        upper = np.nan
    return lower, upper


def _ci_worker(args):
    """Top-level worker for parallel CI computation (must be module-level for pickling)."""
    subrows, ci_type = args
    return get_cis(subrows, ci_type)


def compute_cis_parallel(subrows_list, ci_type, n_workers=None):
    """
    Compute CIs for a list of data slices in parallel using multiprocessing.
    Returns a list of (lower, upper) tuples in the same order as subrows_list.
    """
    if ci_type is None or ci_type == "":
        return [(np.nan, np.nan)] * len(subrows_list)
    if n_workers is None:
        n_workers = min(len(subrows_list), os.cpu_count() or 1)
    args = [(subrows, ci_type) for subrows in subrows_list]
    with Pool(processes=n_workers) as pool:
        results = pool.map(_ci_worker, args)
    return results


################################################################
# Functions to compute and visualize approval scores for different 
# slices of data
################################################################

def get_approval_per_model_stance(data, issue, likert, ci_type="2-bootstrap"):
    """
    Compute mean approval rate per model and model stance, separate for users on each side of the issue.
    data: a dataframe outputted by clean_qualtrics_data()
    """
    assert not data.isna().any().any(), "Data contains NaN values"
    assert ci_type in ["1-bootstrap", "2-bootstrap", "regression", None, ""], f"Invalid ci_type {ci_type}"

    rows = get_relevant_subdf(data, issue, likert)
    groups = []
    subrows_list = []
    for (model, model_stance, user_stance), subrows in rows.groupby(["model", "model_stance", "user_stance"]):
        mean_per_question = subrows.groupby("qid")["user_score"].mean()  # average score per question
        groups.append({
            "model": model,
            "model_stance": model_stance,
            "user_stance": user_stance,
            "num_responses": len(subrows),
            "num_questions": len(mean_per_question),
            "mean_score": mean_per_question.mean(),
            "ci_type": ci_type,
        })
        subrows_list.append(subrows)

    # compute cis in parallel to speed up processing, if needed
    ci_results = compute_cis_parallel(subrows_list, ci_type)
    summary = [{**g, "ci_lower": lo, "ci_upper": hi} for g, (lo, hi) in zip(groups, ci_results)]

    return pd.DataFrame(summary)

def get_approval_per_question_valence(data, issue, likert, ci_type="2-bootstrap", only_default=True):
    """
    Compute mean approval rate per question valence, separate for users on each side of the issue.
    data: a dataframe outputted by clean_qualtrics_data()
    """
    assert not data.isna().any().any(), "Data contains NaN values"
    assert ci_type in ["1-bootstrap", "2-bootstrap", "regression", None, ""], f"Invalid ci_type {ci_type}"
    assert "question_valence" in data.columns, "Expected column question_valence in data"

    rows = get_relevant_subdf(data, issue, likert)
    if only_default:
        rows = rows[rows["model_stance"] == "default"]
        print("Only keeping rows where model_stance = default ->", len(rows), "rows")
    groups = []
    subrows_list = []
    for (question_valence, user_stance), subrows in rows.groupby(["question_valence", "user_stance"]):
        mean_per_question = subrows.groupby("qid")["user_score"].mean()  # average score per question
        groups.append({
            "question_valence": question_valence,
            "user_stance": user_stance,
            "num_responses": len(subrows),
            "num_questions": len(mean_per_question),
            "mean_score": mean_per_question.mean(),
            "ci_type": ci_type,
        })
        subrows_list.append(subrows)
    ci_results = compute_cis_parallel(subrows_list, ci_type)
    summary = [{**g, "ci_lower": lo, "ci_upper": hi} for g, (lo, hi) in zip(groups, ci_results)]
    return pd.DataFrame(summary)

def get_approval_per_question_user_alignment(data, issue, likert, ci_type="2-bootstrap", only_default=True):
    """
    Compute mean approval per Neutral, Aligned, and Misaligned between question valence and user stance.
    data: a dataframe outputted by clean_qualtrics_data()
    """
    assert not data.isna().any().any(), "Data contains NaN values"
    assert ci_type in ["1-bootstrap", "2-bootstrap", "regression", None, ""], f"Invalid ci_type {ci_type}"
    assert "question_valence" in data.columns, "Expected column question_valence in data"
    
    rows = get_relevant_subdf(data, issue, likert).copy()
    if only_default:
        rows = rows[rows["model_stance"] == "default"]
        print("Only keeping rows where model_stance = default ->", len(rows), "rows")

    groups = []
    neutral = rows[rows["question_valence"] == "neutral"]
    groups.append(("Neutral", neutral))

    both_liberal = rows[(rows["user_stance"] == "liberal") & ((rows["question_valence"] == "somewhat liberal") | (rows["question_valence"] == "very liberal"))]
    both_conservative = rows[(rows["user_stance"] == "conservative") & ((rows["question_valence"] == "somewhat conservative") | (rows["question_valence"] == "very conservative"))]
    aligned = pd.concat([both_liberal, both_conservative], ignore_index=True)
    groups.append(("Aligned", aligned))

    liberal_user_conservative_question = rows[(rows["user_stance"] == "liberal") & ((rows["question_valence"] == "somewhat conservative") | (rows["question_valence"] == "very conservative"))]
    conservative_user_liberal_question = rows[(rows["user_stance"] == "conservative") & ((rows["question_valence"] == "somewhat liberal") | (rows["question_valence"] == "very liberal"))]
    misaligned = pd.concat([liberal_user_conservative_question, conservative_user_liberal_question], ignore_index=True)
    groups.append(("Misaligned", misaligned))

    group_meta = []
    subrows_list = []
    for group, subrows in groups:
        mean_per_question = subrows.groupby("qid")["user_score"].mean()  # average score per question
        group_meta.append({
            "group": group,
            "num_responses": len(subrows),
            "num_questions": len(mean_per_question),
            "mean_score": mean_per_question.mean(),
            "ci_type": ci_type,
        })
        subrows_list.append(subrows)
    ci_results = compute_cis_parallel(subrows_list, ci_type)
    summary = [{**g, "ci_lower": lo, "ci_upper": hi} for g, (lo, hi) in zip(group_meta, ci_results)]
    return pd.DataFrame(summary)


def get_approval_by_response_type(
        data, issue, response_order, likert, ci_type="2-bootstrap",
        n_bootstrap=1000, seed=0):
    """
    Returns GPT-only means and CIs by response type and participant side.
    """
    rows = get_relevant_subdf(data, issue, likert).copy()

    # collapse issue-specific stances into shared conservative/liberal sides
    rows["user_stance"] = rows["user_stance_side"]
    rows["model_stance"] = rows["model_stance_side"]
    # only GPT has the side-specific and balanced response types
    rows = rows[rows["model"] == "gpt"]
    rows = rows[rows["model_stance"].isin(response_order)]

    def group_seed(response_type, user_side):
        # stable per-cell seed so shared cells match across plot variants
        key = f"{seed}|{issue}|{likert}|{response_type}|{user_side}"
        return zlib.crc32(key.encode("utf-8"))

    def two_stage_ci_fast(subrows, rng):
        # cache per-question arrays before bootstrapping to avoid repeated pandas slicing
        question_groups = [
            group["user_score"].to_numpy(dtype=float)
            for _, group in subrows.groupby("qid", sort=False)
        ]

        if len(question_groups) <= 1:
            return np.nan, np.nan

        n_questions = len(question_groups)
        bootstrap_means = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            sampled_question_idxs = rng.integers(0, n_questions, size=n_questions)
            question_means = np.empty(n_questions)

            for i, question_idx in enumerate(sampled_question_idxs):
                scores = question_groups[question_idx]
                sampled_score_idxs = rng.integers(0, len(scores), size=len(scores))
                question_means[i] = scores[sampled_score_idxs].mean()

            bootstrap_means[b] = question_means.mean()

        return np.percentile(bootstrap_means, 2.5), np.percentile(bootstrap_means, 97.5)

    summary = []
    for (response_type, user_side), subrows in rows.groupby(["model_stance", "user_stance"]):
        mean_per_question = subrows.groupby("qid")["user_score"].mean()

        if ci_type in [None, ""]:
            ci_lower, ci_upper = np.nan, np.nan
        elif ci_type == "2-bootstrap":
            rng = np.random.default_rng(group_seed(response_type, user_side))
            ci_lower, ci_upper = two_stage_ci_fast(subrows, rng)
        else:
            raise ValueError("ci_type must be 2-bootstrap or empty")

        summary.append({
            "response_type": response_type,
            "user_stance": user_side,
            "num_responses": len(subrows),
            "num_questions": len(mean_per_question),
            "mean_score": mean_per_question.mean(),
            "ci_type": ci_type,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    return pd.DataFrame(summary)


def plot_default_vs_balanced_response_type_grid(
        data, issue="all", likert_num2question=None, metric_labels=None,
        ci_type="2-bootstrap", n_bootstrap=1000, seed=0,
        title=None, y_min=0, y_max=1, save_path=None, show=True):
    """
    Combine the default-middle and balanced-middle response-type plots.
    """
    likerts = [f"likert_{i}" for i in range(1, 6)]
    if metric_labels is None:
        metric_labels = [
            likert_num2question[likert] if likert_num2question is not None else likert
            for likert in likerts
        ]
    # keep full question text, but wrap it so column titles do not collide
    metric_labels = [
        textwrap.fill(label, width=28, break_long_words=False)
        for label in metric_labels
    ]

    # the two rows differ only in the middle response type
    row_specs = [
        ("default", "GPT Default", ["Cons. side", "Default", "Lib. side"]),
        ("chunked", "GPT Balanced", ["Cons. side", "Balanced", "Lib. side"]),
    ]
    colors = {
        "conservative": "#de8f02",
        "liberal": "#0072B2",
    }

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), sharey=True)
    all_summaries = []

    for row_idx, (middle_response_type, row_label, response_labels) in enumerate(row_specs):
        response_order = ["conservative", middle_response_type, "liberal"]

        for col_idx, likert in enumerate(likerts):
            ax = axes[row_idx, col_idx]
            summary = get_approval_by_response_type(
                data, issue, response_order, likert, ci_type=ci_type,
                n_bootstrap=n_bootstrap, seed=seed,
            )
            summary = summary.copy()
            summary["middle_response_type"] = middle_response_type
            summary["likert"] = likert
            all_summaries.append(summary)

            x = np.arange(len(response_order))
            for user_side in ["conservative", "liberal"]:
                side_summary = (
                    summary[summary["user_stance"] == user_side]
                    .set_index("response_type")
                    .reindex(response_order)
                )
                means = side_summary["mean_score"].to_numpy(dtype=float)
                if ci_type in [None, ""]:
                    yerr = None
                else:
                    lower = means - side_summary["ci_lower"].to_numpy(dtype=float)
                    upper = side_summary["ci_upper"].to_numpy(dtype=float) - means
                    yerr = np.array([lower, upper])

                ax.errorbar(
                    x, means, yerr=yerr, marker="o", linewidth=2,
                    capsize=4, label=f"{user_side.capitalize()} side",
                    color=colors[user_side],
                )

            if row_idx == 0:  # only title the top row
                ax.set_title(metric_labels[col_idx], fontsize=10, pad=8, weight="normal")
            ax.set_xticks(x)
            ax.set_xticklabels(response_labels, fontsize=9)
            ax.set_ylim(y_min, y_max)
            ax.grid(axis="y", alpha=0.3)
            if col_idx != 0:  # only show y tick labels on the leftmost column
                ax.tick_params(labelleft=False)

        axes[row_idx, 0].text(
            -0.28, 0.5, row_label, transform=axes[row_idx, 0].transAxes,
            ha="right", va="center", fontsize=11, weight="normal",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=10)
    # keep the global x label above the legend so the two do not overlap
    fig.text(0.5, 0.095, "Response Type", ha="center", fontsize=11)
    fig.text(0.04, 0.5, "Mean Approval Score", va="center",
             rotation="vertical", fontsize=11)

    if title is not None:
        fig.suptitle(title, y=0.98, fontsize=14)
    plt.tight_layout(rect=[0.06, 0.14, 1, 0.94 if title is not None else 0.98])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"Saved default vs balanced response-type grid to {save_path}")
    if show:
        plt.show()

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    return fig, axes, combined_summary


def get_plot_title(issue, likert, likert_num2question=None):
    """
    Short function shared across visualizations to get appropriate title based on issue and Likert question.
    """
    if issue == "all":
        title = "All Issues"
    else:
        title = issue.replace("_", " ").capitalize()
    if likert == "all":       
        title += "\nAveraged over approval questions"
    else:
        if likert_num2question is not None:  # map to question text
            title += "\n" + likert_num2question[likert]
        else:
            title += f"\nLikert {likert}"
        if likert in REVERSE_LIKERTS:
            title += " (reversed)"
    return title

def _compute_loss(x, y, model_stances):
    """
    Compute loss of the balanced (chunked) point vs. the best achievable single-axis scores.
    Returns (ref_x, ref_y, max_x, max_y), or (None, None, None, None) if not enough data.
    """
    if len(x) < 2:
        return None, None, None, None

    ref_idx = next(i for i, ms in enumerate(model_stances) if ms == "chunked")
    ref_x = x[ref_idx]
    ref_y = y[ref_idx]

    # max x and y from among single-sided models
    single_sided = [i for i, ms in enumerate(model_stances) if ms not in ("default", "chunked")]
    max_x = max(x[i] for i in single_sided) if single_sided else max(x)
    max_y = max(y[i] for i in single_sided) if single_sided else max(y)
    return ref_x, ref_y, max_x, max_y


def compute_losses_all_issues(issue2summary, min_responses=3, min_questions=3):
    """
    Compute conservative and liberal loss for every issue.
    Conservative loss (x-axis) and liberal loss (y-axis) are expressed as percentages
    of the best achievable single-side score. The reference point is the chunked
    (balanced) model.
    Returns a DataFrame with columns: issue, loss_conservative, loss_liberal, loss_total.
    """
    rows = []
    for issue, summary in issue2summary.items():
        x_stance, y_stance, _ = STANCE_DICT[issue]
        x, y, model_stances = [], [], []
        for (model, model_stance), subrows in summary.groupby(["model", "model_stance"]):
            if len(subrows) != 2:
                continue
            if subrows["num_responses"].min() < min_responses:
                continue
            if subrows["num_questions"].min() < min_questions:
                continue
            subrows = subrows.set_index("user_stance")
            x.append(subrows.loc[x_stance, "mean_score"])
            y.append(subrows.loc[y_stance, "mean_score"])
            model_stances.append(model_stance)
        ref_x, ref_y, max_x, max_y = _compute_loss(x, y, model_stances)
        pct_cons = abs(ref_x - max_x) / max_x * 100 if max_x else 0.0
        pct_lib  = abs(ref_y - max_y) / max_y * 100 if max_y else 0.0
        rows.append({
            "issue": issue,
            "loss_conservative": pct_cons,
            "loss_liberal": pct_lib,
            "loss_total": pct_cons + pct_lib,
        })
    return pd.DataFrame(rows)


def _plot_loss_bars(ax, x, y, model_stances=None):
    """
    Plot loss bars showing how far the balanced (chunked) point is from the best achievable
    single-axis scores. Draws a red arrow along the bottom (conservative/x-loss) and a blue
    arrow along the left edge (liberal/y-loss) of the axes.
    """
    def _format_pct_label(value):
        rounded = round(value)
        if rounded > 0:
            return f"+{rounded}%"
        if rounded < 0:
            return f"{rounded}%"
        return "0%"

    if len(x) < 2:
        return

    ref_x, ref_y, max_x, max_y = _compute_loss(x, y, model_stances)
    if ref_x is None:
        return

    # signed percentages: negative means the chunked model underperforms the best single-sided
    signed_pct_x = (ref_x - max_x) / max_x * 100 if max_x != 0 else 0.0
    signed_pct_y = (ref_y - max_y) / max_y * 100 if max_y != 0 else 0.0

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]

    # X-loss bar: sits on the bottom axis; arrow tail at max_x, head at ref_x
    bary = ylim[0]
    if round(signed_pct_x) == 0:
        ax.plot(ref_x, bary, "o", color="red", markersize=4, clip_on=False)
        mid_x = ref_x
    else:
        ax.annotate("", xy=(ref_x, bary), xytext=(max_x, bary),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5,
                                    mutation_scale=10, shrinkA=0, shrinkB=0),
                    annotation_clip=False)
        mid_x = (ref_x + max_x) / 2
    ax.text(mid_x, bary + yrange * 0.015, _format_pct_label(signed_pct_x),
            ha="center", va="bottom", color="red", fontsize=8)

    # Y-loss bar: sits on the left axis; arrow tail at max_y, head at ref_y
    barx = xlim[0]
    if round(signed_pct_y) == 0:
        ax.plot(barx, ref_y, "o", color="blue", markersize=4, clip_on=False)
        mid_y = ref_y
    else:
        ax.annotate("", xy=(barx, ref_y), xytext=(barx, max_y),
                    arrowprops=dict(arrowstyle="-|>", color="blue", lw=1.5,
                                    mutation_scale=10, shrinkA=0, shrinkB=0),
                    annotation_clip=False)
        mid_y = (ref_y + max_y) / 2
    ax.text(barx + xrange * 0.015, mid_y, _format_pct_label(signed_pct_y),
            ha="left", va="center", color="blue", fontsize=8)


def make_model_stance_scatter_plot(summary, issue, likert, min_responses=3, min_questions=3, 
        likert_num2question=None, include_cis=True, ax=None, zoom=False, ax_min=0, ax_max=1, 
        title=None, x_label=None, y_label=None, label_defaults=True, save_path=None, plot_loss_bars=False, show_legend=True):
    """
    Make scatter plot for this issue. Each dot is a model/model_stance combination, with x/y coordinates corresponding to 
    average user score for users on each side of the issue.
    
    summary: dataframe from compute_means_and_cis with columns model, model_stance, user_stance, num_responses, 
    mean_score, ci_lower, ci_upper
    """
    assert issue in STANCE_DICT or issue == "all", f"Issue {issue} not found in data"
    assert likert in ["all"] + LIKERT_EXT, f"Invalid likert {likert}"
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    
    # always put conservative side on x, liberal side on y
    if issue == "all":
        x_stance = "conservative"
        y_stance = "liberal"
        if x_label is None:
            x_label = "Participants on conservative side"
        if y_label is None:
            y_label = "Participants on liberal side"
    else:
        x_stance, y_stance, _ = STANCE_DICT[issue]
        # have max 10 words per line
        if x_label is None:
            x_label = x_stance.split()
            x_label = "\n".join([" ".join(x_label[i:i+10]) for i in range(0, len(x_label), 10)])
        if y_label is None:
            y_label = y_stance.split()
            y_label = "\n".join([" ".join(y_label[i:i+10]) for i in range(0, len(y_label), 10)])
    
    stances = summary["user_stance"].unique()
    assert set(stances) == {x_stance, y_stance}, f"Expected stances to be {x_stance} and {y_stance}, but got {stances}"
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)

    if title is None:
        title = get_plot_title(issue, likert, likert_num2question)
    ax.set_title(title, fontsize=11)

    x = []
    y = []
    colors = []
    labels = []
    model_stances_list = []
    x_cis = []
    y_cis = []
    for (model, model_stance), subrows in summary.groupby(["model", "model_stance"]):
        if len(subrows) != 2:
            print(f"Warning: {model}, {model_stance} does not have responses from both stances")
            continue
        if subrows["num_responses"].min() < min_responses:
            print(f"Warning: {model}, {model_stance} does not have at least {min_responses} responses for both stances")
            continue
        if subrows["num_questions"].min() < min_questions:
            print(f"Warning: {model}, {model_stance} does not have responses to at least {min_questions} questions for both stances")
            continue
        subrows = subrows.set_index("user_stance")
        if model_stance == "default":
            if label_defaults:
                labels.append(model)
            else:
                labels.append("")
        elif model_stance == "chunked":
            if label_defaults:
                labels.append(f"{model}, balanced")
            else:
                labels.append("balanced")
        elif model_stance == "liberal":
            if label_defaults:
                labels.append(f"{model}, lib. side")
            else:
                labels.append("lib. side")
        elif model_stance == "conservative":
            if label_defaults:
                labels.append(f"{model}, cons. side")
            else:
                labels.append("cons. side")
        elif model_stance == "for":
            if label_defaults:
                labels.append(f"{model}, for")
            else:
                labels.append("for")
        elif model_stance == "against":
            if label_defaults:
                labels.append(f"{model}, against")
            else:
                labels.append("against")
        else:
            print(f"Warning: unexpected model stance {model_stance} for model {model}")
            labels.append(model + ", " + model_stance)
        colors.append(MODEL_COLORS.get(model, "black"))
        model_stances_list.append(model_stance)
        x.append(subrows.loc[x_stance, "mean_score"])
        if x[-1] < ax_min or x[-1] > ax_max:
            print(f"Warning: {model}, {model_stance} has mean score {x[-1]:.4f} for stance {x_stance}, which is outside of axis limits [{ax_min}, {ax_max}]")
        y.append(subrows.loc[y_stance, "mean_score"])
        if y[-1] < ax_min or y[-1] > ax_max:
            print(f"Warning: {model}, {model_stance} has mean score {y[-1]:.4f} for stance {y_stance}, which is outside of axis limits [{ax_min}, {ax_max}]")
        if include_cis:
            x_cis.append((subrows.loc[x_stance, "ci_lower"], subrows.loc[x_stance, "ci_upper"]))
            y_cis.append((subrows.loc[y_stance, "ci_lower"], subrows.loc[y_stance, "ci_upper"]))

        # check that data matches what we expect based on model stance
        if issue == "all":
            assert model_stance in ["default", "chunked", "liberal", "conservative"]
            if model_stance == "liberal" and x[-1] > y[-1]:
                print(f"Warning: unexpected result, liberal-aligned model has higher score among users on conservative side")
            if model_stance == "conservative" and y[-1] > x[-1]:
                print(f"Warning: unexpected result, conservative-aligned model has higher score among users on liberal side")
        else:
            assert model_stance in ["default", "chunked", "for", "against"]
            for_side = STANCE_DICT[issue][2]
            liberal_side = "for" if for_side == "liberal" else "against"
            conservative_side = "for" if for_side == "conservative" else "against"
            if model_stance == liberal_side and x[-1] > y[-1]:
                print(f"Warning: unexpected result, liberal-aligned model ({model_stance}) has higher score among users on conservative side")
            if model_stance == conservative_side and y[-1] > x[-1]:
                print(f"Warning: unexpected result, conservative-aligned model ({model_stance}) has higher score among users on liberal side")
    
    if include_cis:
        for i in range(len(labels)):
            xerr = [[x[i] - x_cis[i][0]], [x_cis[i][1] - x[i]]]
            yerr = [[y[i] - y_cis[i][0]], [y_cis[i][1] - y[i]]]
            ax.errorbar(x[i], y[i], xerr=xerr, yerr=yerr, fmt='o',
                        color=colors[i], ecolor='#bbbbbb', elinewidth=0.8,
                        capsize=0, zorder=2, markersize=6)
    else:
        ax.scatter(x, y, c=colors, zorder=2, edgecolors='none')
    for i, label in enumerate(labels):
        # offset labels by 0.001 in x and y direction to avoid overlap with points
        ax.annotate(label, (x[i] + 0.001, y[i] + 0.001), fontsize=9)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    # add legend if we haven't labeled points directly with model names
    if not label_defaults and show_legend:
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=model, markerfacecolor=color, markersize=8) for model, color in MODEL_COLORS.items()]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    if zoom:
        # zoom in as much as the data allows
        if include_cis and not any(np.isnan(ci[0]) or np.isnan(ci[1]) for ci in x_cis):
            min_x = min([ci[0] for ci in x_cis])
            max_x = max([ci[1] for ci in x_cis])
        else:
            min_x = min(x)
            max_x = max(x)
        if include_cis and not any(np.isnan(ci[0]) or np.isnan(ci[1]) for ci in y_cis):
            min_y = min([ci[0] for ci in y_cis])
            max_y = max([ci[1] for ci in y_cis])
        else:
            min_y = min(y)
            max_y = max(y)
        min_both = min(min_x, min_y)
        max_both = max(max_x, max_y)
        ax.set_xlim(min_both - 0.02, max_both + 0.02)
        ax.set_ylim(min_both - 0.02, max_both + 0.02)
    else:
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
    ax.grid(alpha=0.3)
    if plot_loss_bars:
        _plot_loss_bars(ax, x, y, model_stances=model_stances_list)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"Saved scatter plot to {save_path}")


def make_scatter_plots_for_all_issues(issue2summary, likert, likert_num2question, ax_min=0, ax_max=1, plot_loss_bars=False, save_path=None):
    """
    Make scatter plots for all issues, arranged in a grid.
    """
    assert set(issue2summary.keys()) == set(STANCE_DICT.keys()), f"Expected issues {set(STANCE_DICT.keys())}, but got {set(issue2summary.keys())}"
    num_rows = 5
    num_cols = math.ceil(len(issue2summary) / num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3*num_rows), sharex=True, sharey=True)
    fig.suptitle(f"Approval question: {likert_num2question[likert]}", fontsize=14)
    
    for i, issue in enumerate(issue2summary.keys()):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        title = issue.replace("_", " ").capitalize()
        if row == num_rows - 1:  # only label x-axis on bottom row
            x_label = "Conservative side"
        else:            
            x_label = ""
        if col == 0:  # only label y-axis on leftmost column
            y_label = "Liberal side"
        else:            
            y_label = ""   
    
        show_legend = (row == 0 and col == num_cols - 1)
        make_model_stance_scatter_plot(issue2summary[issue], issue, likert, ax=ax, include_cis=False, 
            ax_min=ax_min, ax_max=ax_max, x_label=x_label, y_label=y_label, title=title, 
            label_defaults=False, plot_loss_bars=plot_loss_bars, show_legend=show_legend)
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.1, top=0.93)  # top leaves room for suptitle
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"Saved scatter plot grid to {save_path}")
    plt.show(fig)


def plot_approval_by_valence(summary, issue, likert, likert_num2question=None, model2summary=None,
                             ax=None, title=None, include_xlabels=True, include_ylabels=True,
                             save_path=None):
    """
    Plot approval by question valence, with separate lines for users on each side.
    summary: dataframe from get_approval_per_question_valence with columns question_valence, user_stance, num_responses, mean_score, ci_lower, ci_upper
    """
    valences = ["very conservative", "somewhat conservative", "neutral", "somewhat liberal", "very liberal"]
    assert len(summary) == (len(valences) * 2), f"Expected summary length {len(valences) * 2}, but got {len(summary)}"
    user_conservative = summary[summary["user_stance"] == "conservative"].set_index("question_valence")
    user_liberal = summary[summary["user_stance"] == "liberal"].set_index("question_valence")
    x = np.arange(len(valences))
    y1 = [user_conservative.loc[valence, "mean_score"] for valence in valences]
    y1_cis = [(user_conservative.loc[valence, "ci_lower"], user_conservative.loc[valence, "ci_upper"]) for valence in valences]
    y2 = [user_liberal.loc[valence, "mean_score"] for valence in valences]
    y2_cis = [(user_liberal.loc[valence, "ci_lower"], user_liberal.loc[valence, "ci_upper"]) for valence in valences]   

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y1, label="Cons. Side", color="red")
    ax.fill_between(x, [y1_cis[i][0] for i in range(len(x))], [y1_cis[i][1] for i in range(len(x))], color="red", alpha=0.1)

    ax.plot(x, y2, label="Lib. Side", color="blue")
    ax.fill_between(x, [y2_cis[i][0] for i in range(len(x))], [y2_cis[i][1] for i in range(len(x))], color="blue", alpha=0.1)
    
    if model2summary is not None:
        # overlay dots per model
        for model, model_summary in model2summary.items():
            assert len(model_summary) == (len(valences) * 2), f"Expected model_summary length {len(valences) * 2}, but got {len(model_summary)} for model {model}"
            model_user_conservative = model_summary[model_summary["user_stance"] == "conservative"].set_index("question_valence")
            model_user_liberal = model_summary[model_summary["user_stance"] == "liberal"].set_index("question_valence")
            model_y1 = [model_user_conservative.loc[valence, "mean_score"] for valence in valences]
            model_y2 = [model_user_liberal.loc[valence, "mean_score"] for valence in valences]
            ax.scatter(x, model_y1, color=MODEL_COLORS.get(model, "black"), marker="x")
            ax.scatter(x, model_y2, color=MODEL_COLORS.get(model, "black"), marker="o")
    ax.set_xticks(x)
    if title is None:
        title = "Default AI Responses, " + get_plot_title(issue, likert, likert_num2question)
    ax.set_title(title)
    if include_xlabels:
        valence_labels = ["Very Cons.", "Somewhat Cons.", "Neutral", "Somewhat Lib.", "Very Lib."]
        ax.set_xticklabels(valence_labels, fontsize=10)
        ax.set_xlabel("Prompt Stance", fontsize=11)
    if include_ylabels:
        ax.set_ylabel("Mean Approval Score", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"Saved approval by valence plot to {save_path}")

def plot_approval_by_user_alignment(summary, issue, likert, likert_num2question=None,
                                    ax=None, title=None, include_xlabels=True, include_ylabels=True,
                                    save_path=None, y_min=None, y_max=None):
    """
    Plot approval by user alignment (Neutral vs. Aligned vs. Misaligned).
    summary: dataframe from get_approval_per_question_user_alignment with columns group, num_responses, mean_score, ci_lower, ci_upper
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    alignments = ["Neutral", "Aligned", "Misaligned"]
    summary = summary.set_index("group")

    x = np.arange(3)
    means = summary.loc[alignments, "mean_score"].values
    ax.scatter(x, means, marker="o")
    upper_diff = summary.loc[alignments, "ci_upper"].values - means
    lower_diff = means - summary.loc[alignments, "ci_lower"].values
    yerr = np.array([lower_diff, upper_diff])
    ax.errorbar(x, means, yerr=yerr, fmt="none", capsize=5)
    if include_xlabels:
        ax.set_xticks(x)
        ax.set_xticklabels(["Neutral", "Charged,\nAligned", "Charged,\nMisaligned"], fontsize=10)
    if include_ylabels:
        ax.set_ylabel("Mean approval score", fontsize=11)
    if title is None:
        title = "Comparing Neutral and Charged Prompts\nDefault AI Responses, " + get_plot_title(issue, likert, likert_num2question)
    ax.set_title(title, fontsize=11)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)


def get_stats_per_metric_and_model_stance(issue2summary):
    """
    Get table with stats per issue, model, model stance, and metric.
    issue2summary: a dict mapping issue to a summary, which is output from get_approval_per_model_stance for that issue
    """
    assert set(issue2summary.keys()) == set(STANCE_DICT.keys()), f"Issues in issue2summary do not match issues in STANCE_DICT"
    balance_metrics = ["diff", "ratio", "min_score"]
    metrics = ["conservative_score", "liberal_score"] + balance_metrics
    stats = []
    for issue, issue_df in issue2summary.items():
        issue_stats = []
        for (model, model_stance), subrows in issue_df.groupby(["model", "model_stance"]):
            if len(subrows) != 2:
                print(f"Warning: {model}, {model_stance} does not have responses from both stances for issue {issue}")
                continue
            conservative_score = subrows[subrows["user_stance"] == STANCE_DICT[issue][0]]["mean_score"].values[0]
            liberal_score = subrows[subrows["user_stance"] == STANCE_DICT[issue][1]]["mean_score"].values[0]
            diff = conservative_score - liberal_score
            ratio = np.log(conservative_score / liberal_score) if liberal_score > 0 else np.inf
            min_score = min(conservative_score, liberal_score)

            # map model stance to side
            if model_stance == "default" or model_stance == "chunked":
                model_stance_side = model_stance
            elif model_stance == "for":
                model_stance_side = STANCE_DICT[issue][2]  # which side aligns with model stance = "for"
            elif model_stance == "against":
                model_stance_side = "conservative" if STANCE_DICT[issue][2] == "liberal" else "liberal"
            else:
                print(f"Warning: unexpected model stance {model_stance} for issue {issue}")
                model_stance_side = None

            issue_stats.append({
                "issue": issue,
                "model": model,
                "model_stance": model_stance_side,
                "conservative_score": conservative_score,
                "liberal_score": liberal_score,
                "diff": diff,
                "diff_abs": abs(diff),
                "ratio": ratio,
                "ratio_abs": abs(ratio),
                "min_score": min_score,
            })
        issue_stats = pd.DataFrame(issue_stats)

        pareto = []
        # mark whether each model + model_stance is Pareto optimal
        for i, row_i in issue_stats.iterrows():
            dominated = False
            for j, row_j in issue_stats.iterrows():
                if i == j:
                    continue
                if (row_j["conservative_score"] > row_i["conservative_score"] and row_j["liberal_score"] >= row_i["liberal_score"]):
                    dominated = True
                    break
                if (row_j["conservative_score"] >= row_i["conservative_score"] and row_j["liberal_score"] > row_i["liberal_score"]):
                    dominated = True
                    break
            pareto.append(not dominated)
        issue_stats["pareto_optimal"] = pareto

        for metric in metrics:
            if metric == "ratio":  # closest to 0 wins
                winner = np.argmin(issue_stats["ratio_abs"].values)
            elif metric == "diff":  # closest to 0 wins
                winner = np.argmin(issue_stats["diff_abs"].values)  
            else:  # max wins
                winner = np.argmax(issue_stats[metric].values)  
            issue_stats[metric + "_winner"] = False
            issue_stats.loc[winner, metric + "_winner"] = True

        for balance_metric in balance_metrics:
            # find MEA among Pareto optimal points
            pareto_points = issue_stats[issue_stats["pareto_optimal"]]
            if balance_metric == "ratio":
                mea = np.argmin(pareto_points["ratio_abs"].values)
            elif balance_metric == "diff":
                mea = np.argmin(pareto_points["diff_abs"].values)
            else:
                assert balance_metric == "min_score"
                mea = np.argmax(pareto_points[balance_metric].values)  
            issue_stats[balance_metric + "_mea"] = False
            mea_index = pareto_points.index[mea]
            issue_stats.loc[mea_index, balance_metric + "_mea"] = True

        stats.append(issue_stats)
    
    cols = ["pareto_optimal"]
    for metric in metrics:
        cols.append(metric) 
        cols.append(metric + "_winner")
        if metric in balance_metrics:
            cols.append(metric + "_mea")
    stats = pd.concat(stats, ignore_index=True)
    return stats[["issue", "model", "model_stance"] + cols]

##############
# Regression
##############

def get_variable_type(name, return_groups=False):
    """
    Categorize variable names into blocks/groups.
    """
    if name == "const":
        return "intercept"
    if name in LIKERT_EXT:
        return "likert"
    for issue in STANCE_DICT:
        if name == f"issue_{issue}":
            return "issue"
    for col in ["Age", "Sex", "Ethnicity simplified", "Student status", "Employment status"]:
        if name.startswith(col):
            if return_groups: 
                return col.split("_")[0]
            else:
                return "demographics"                
    for col in ["neutral", "somewhat charged", "very charged"]:
        if name == col:
            return "question charge"
    return "model x model_stance x user_stance"

def get_regression_inputs(data):
    """
    Prepare data for regression. Inputs:
      - model x model_stance_side x user_stance (full interaction dummies, ref: claude x default x conservative)
      - question charge (dummies for somewhat charged and very charged, ref: neutral)
      - issue (dummies, ref: issue_abortion)
      - likert question (dummies, ref: first question)
      - demographic variables: age, sex, ethnicity, student_status, employment_status (ref: missing values)
    Clustered standard errors at the question (qid) and participant (prolific_id) level.
    """
    rows = data.copy()
    rows["user_stance"] = rows["user_stance_side"]  # we're analyzing multiple issues at once, so group user stance by side
    rows["model_stance"] = rows["model_stance_side"]  # we're analyzing multiple issues at once, so group model stance by side
    group2ref = {}  # track reference cell for each group of variables for easier interpretation of regression output

    # ----------------------------------------------------------------
    # 1. model x model_stance_side x user_stance dummies
    #    reference cell: alphabetical first
    # ----------------------------------------------------------------
    assert rows[["model", "model_stance", "user_stance"]].isna().sum().sum() == 0, "Expected no NaNs in model, model_stance, or user_stance columns"
    model_stance_user = pd.get_dummies(
        rows[["model", "model_stance", "user_stance"]]
        .apply(lambda r: f"{r['model']}_{r['model_stance']}_{r['user_stance']}", axis=1),
        prefix="", prefix_sep=""
    ).astype(int)
    ref_cell = sorted(model_stance_user.columns)[0]
    model_stance_user = model_stance_user.drop(columns=[ref_cell])
    group2ref["model x model_stance x user_stance"] = ref_cell

    # ----------------------------------------------------------------
    # 2. question charge dummies
    #    reference cell: neutral
    # ----------------------------------------------------------------
    assert "question_valence" in rows.columns, "Expected column question_valence in data"
    assert rows["question_valence"].isna().sum() == 0, "Expected no NaNs in question_valence column"
    question_charge = np.zeros((len(rows), 2))  # Neutral (reference), Somewhat charged, Very charged
    for i, r in rows.reset_index(drop=True, inplace=False).iterrows():
        if r["question_valence"] == "neutral":
            continue  # neutral as reference
        elif r["question_valence"] in ["somewhat liberal", "somewhat conservative"]:
            question_charge[i, 0] = 1  # somewhat charged
        else:
            assert r["question_valence"] in ["very liberal", "very conservative"]
            question_charge[i, 1] = 1  # very charged
    question_charge = pd.DataFrame(question_charge, columns=["somewhat charged", "very charged"]).set_index(rows.index)
    print(f"Question charge: {question_charge['somewhat charged'].mean():.3f} somewhat charged, {question_charge['very charged'].mean():.3f} very charged")
    group2ref["question charge"] = "neutral"

    # ----------------------------------------------------------------
    # 3. issue dummies
    #    reference cell: alphabetical first
    # ----------------------------------------------------------------
    assert rows["issue"].isna().sum() == 0, "Expected no NaNs in issue column"
    issue_dummies = pd.get_dummies(rows["issue"], prefix="issue", prefix_sep="_").astype(int)
    ref_cell = sorted(issue_dummies.columns)[0]
    issue_dummies = issue_dummies.drop(columns=[ref_cell])
    group2ref["issue"] = ref_cell

    # ----------------------------------------------------------------
    # 4. likert dummies
    #    reference cell: alphabetical first
    # ----------------------------------------------------------------
    assert rows["likert"].isna().sum() == 0, "Expected no NaNs in issue column"
    likert_dummies = pd.get_dummies(rows["likert"], prefix="", prefix_sep="").astype(int)
    ref_cell = sorted(likert_dummies.columns)[0]
    likert_dummies = likert_dummies.drop(columns=[ref_cell])
    group2ref["likert"] = ref_cell

    # ----------------------------------------------------------------
    # 5. demographic variables with missingness indicators
    #    reference cell: missing values 
    # ----------------------------------------------------------------
    demo_parts = []
    demo_cols = {
        'Age': 'continuous',
        'Sex': 'categorical',
        'Ethnicity simplified': 'categorical',
        'Student status': 'categorical',
        'Employment status': 'categorical'
    }
    for col, col_type in demo_cols.items():
        assert col in rows.columns, f"Expected demographic column '{col}' in data"
        if col_type == "continuous":
            rows[col] = pd.to_numeric(rows[col], errors="coerce")  # convert to float, non-numeric -> NaN
            col_mean = rows[col].mean()
            print(f"Warning: filling {rows[col].isna().mean():.4f} missing values in continuous demographic column '{col}' with mean value {col_mean:.3f}")
            rows[col] = rows[col].fillna(col_mean)  # impute missing values
            demo_parts.append(rows[col].to_frame())
        else:
            assert col_type == "categorical", f"Expected col_type 'categorical' or 'continuous' for column '{col}', but got '{col_type}'"
            missing = (rows[col] == "Prefer not to say") | (rows[col] == "CONSENT_REVOKED") | (rows[col] == "DATA_EXPIRED")
            rows.loc[missing, col] = np.nan  # fill missing values with NaN
            dummies = pd.get_dummies(rows[col], prefix=col).astype(int)  # leave missing values as 0's, treat as reference
            # missing rows are the implicit reference category
            assert rows[col].isna().any(), f"No missing values for {col}; need to drop one dummy category or the model will be collinear with the intercept"
            demo_parts.append(dummies)
        print(f"Demographic column '{col}': {rows[col].isna().mean():.3f} missing values")
    demo_df = pd.concat(demo_parts, axis=1)
    group2ref["demographics"] = "missing"

    # ----------------------------------------------------------------
    # 6. Assemble design matrix and fit
    # ----------------------------------------------------------------
    X = pd.concat([model_stance_user, question_charge, issue_dummies, likert_dummies, demo_df], axis=1)
    X = sm.add_constant(X)
    X = X.astype(float)
    y = rows["user_score"].astype(float)
    # cov_kwds = {"groups": rows[["qid", "prolific_id"]]}
    clusters = pd.DataFrame({
        "qid": pd.Categorical(rows["qid"]).codes,
        "prolific_id": pd.Categorical(rows["prolific_id"]).codes,
    })
    return X, y, clusters, group2ref

def fit_regression(X, y, clusters): 
    """
    Fit regression with clustered standard errors.
    """
    model = sm.OLS(y, X)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": clusters.to_numpy()})
    print(result.summary())
    return result


def plot_main_interaction_coefs(result, ref_col, offset=0.1, step_size=1):
    """
    Plot main interaction coefficients from regression. Use special logic to plot 
    conservative and liberal sides next to each other per model / model stance.
    """
    coefs = result.params
    cis = result.conf_int()
    terms = [ref_col] + [t for t in coefs.index if get_variable_type(t) == "model x model_stance x user_stance"]
    # sort into default, conservative, liberal, chunked
    stance_to_order = {"default": 0, "conservative": 1, "liberal": 2, "chunked": 3}
    terms = sorted(terms, key=lambda t: (stance_to_order[t.split("_")[1]], t.split("_")[0]), reverse=True)  # sort by model_stance, then model

    plt.figure(figsize=(5.5,6))
    y_pos = 0
    y_ticks = []
    y_labels = []
    stance_beginning = {}
    for t in terms:
        model, model_stance, user_stance = t.split("_")
        if user_stance == "liberal":
            continue 
        assert user_stance == "conservative", f"Expected term {t} to end with 'conservative'"
        
        if t != ref_col:
            term1 = t
            assert term1 in coefs.index, f"Expected term {term1} in regression results"
            coef = coefs[term1]
            ci_low, ci_high = cis.loc[term1]
            pos = y_pos + offset  # conservative above
            plt.plot([ci_low, ci_high], [pos, pos], color="red", alpha=0.5)
            if y_pos == 0:
                plt.scatter([coef], [pos], color="red", zorder=3, s=30, marker="o", label="Participants on cons. side")
            else:
                plt.scatter([coef], [pos], color="red", zorder=3, s=30, marker="o")

        term2 = f"{model}_{model_stance}_liberal"
        if term2 != ref_col:
            assert term2 in coefs.index, f"Expected term {term2} in regression results"
            coef = coefs[term2]
            ci_low, ci_high = cis.loc[term2]
            pos = y_pos - offset  # liberal below
            plt.plot([ci_low, ci_high], [pos, pos], color="blue", alpha=0.5)
            if y_pos == 0:
                plt.scatter([coef], [pos], color="blue", zorder=3, s=30, marker="o", label="Participants on lib. side")
            else:
                plt.scatter([coef], [pos], color="blue", zorder=3, s=30, marker="o")

        if model_stance == "default":
            y_labels.append(f"{model}, default")
        elif model_stance == "conservative":
            y_labels.append(f"{model}, cons. side")
        elif model_stance == "liberal":
            y_labels.append(f"{model}, lib. side")
        else:
            assert model_stance == "chunked", f"Expected model_stance to be one of 'default', 'liberal', 'conservative', or 'chunked', but got {model_stance}"
            y_labels.append(f"{model}, balanced")
        if model_stance not in stance_beginning:
            stance_beginning[model_stance] = y_pos
        y_ticks.append(y_pos)
        y_pos += step_size

    # add horizontal lines separating stances
    single_start = stance_beginning["conservative"]
    chunked_start = stance_beginning["chunked"]
    plt.axhline(single_start + step_size/2, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    plt.axhline(chunked_start + step_size/2, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    # add y labels
    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Coefficient estimate", fontsize=11)
    ref_model, ref_model_stance, ref_user_stance = ref_col.split("_")
    plt.title(f"Regression coefficients for model x model stance x participant side\nReference cell: {ref_model}, {ref_model_stance}, participant={ref_user_stance}", fontsize=11.5)
    plt.grid(axis="x", alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)


def plot_regression_coefs(result, block2ref, likert_num2question,
                          blocks_to_include=None, skip_const=True, figsize=None):
    """
    Plot regression coefficients with 95% CIs, grouped by variable block.
    """
    coefs = result.params
    cis = result.conf_int()

    names = [n for n in coefs.index if not (skip_const and n == "const")]
    blocks = [get_variable_type(n, return_groups=False) for n in names]

    if blocks_to_include is not None:
        # only keep variables in specified blocks
        names = [n for n, b in zip(names, blocks) if b in blocks_to_include]
        blocks = [b for b in blocks if b in blocks_to_include]

    block_order = [
        "model x model_stance x user_stance",
        "question charge",
        "likert",
        "issue",
        "demographics",
        "intercept"
    ]

    # sort by block then name
    combined = sorted(zip(names, blocks),
                      key=lambda x: (block_order.index(x[1]) if x[1] in block_order else 99, x[0]),
                      reverse=True)
    names_sorted = [c[0] for c in combined]
    blocks_sorted = [c[1] for c in combined]

    if figsize is None:
        figsize = (8, max(6, len(names_sorted) * 0.35))
    
    _, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(len(names_sorted))
    # plot coefficients and CIs 
    for i, (name, block) in enumerate(zip(names_sorted, blocks_sorted)):
        coef = coefs[name]
        ci_low, ci_high = cis.loc[name]
        ax.scatter([coef], [i], color="black", s=20, marker="o")
        ax.plot([ci_low, ci_high], [i, i], color="black", alpha=0.5)

    # add horizontal separator lines between blocks and reference cell for each block
    block_starts = {}
    block_ends = {}
    for i, block in enumerate(blocks_sorted):
        if block not in block_starts:
            block_starts[block] = i
        block_ends[block] = i
    print(f"Block starts: {block_starts}")
    print(f"Block ends: {block_ends}")

    ordered_present_blocks = [
        b for b in block_order if b in block_starts
    ]
    print(f"Ordered present blocks: {ordered_present_blocks}")

    for j, block in enumerate(ordered_present_blocks):
        start = block_starts[block]
        end = block_ends[block]
        # horizontal separator line between blocks
        sep_y = start - 0.5
        ax.axhline(sep_y, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

        # reference text
        if block in block2ref:
            ref_text = f"Reference: {block2ref[block]}"
            center_y = (start + end) / 2
            ax.text(1.02, center_y, ref_text, transform=ax.get_yaxis_transform(),
                    va="center", color="dimgray", rotation=90, fontsize=9)

    ax.axvline(0, color="black", linestyle="--", alpha=0.4)  # include 0 here since it is sometimes meaningful
    ax.set_yticks(y_positions)
    ylabels = []
    if likert_num2question is not None:
        ylabels = [likert_num2question[n] if n in likert_num2question else n for n in names_sorted]
    else:
        ylabels = names_sorted
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Coefficient estimate")
    ax.set_title("Regression coefficients with 95% CIs")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
