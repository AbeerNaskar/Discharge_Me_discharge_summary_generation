import json
import os
from datetime import datetime

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_DIR = "./train"
TEST_DIR = "./test"
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"

DATE_FORMAT = "%d-%m-%Y, %H:%M:%S"

# Evidence keys inside each state
EVIDENCE_KEYS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]

# ── Column definitions ─────────────────────────────────────────────────────────
TRAIN_COLUMNS = [
    "timeline_id",
    "post_id",
    "date",
    "date_number",
    "post_index",
    "Switch",
    "Escalation",
    "Well-being",
    "Adaptive-status",
    "Maladaptive-status",
    "Adaptive-Present",
    "Maladaptive-Present",
    "Adaptive-A-Category",
    "Adaptive-A-highlighted_evidence",
    "Adaptive-B-O-Category",
    "Adaptive-B-O-highlighted_evidence",
    "Adaptive-B-S-Category",
    "Adaptive-B-S-highlighted_evidence",
    "Adaptive-C-O-Category",
    "Adaptive-C-O-highlighted_evidence",
    "Adaptive-C-S-Category",
    "Adaptive-C-S-highlighted_evidence",
    "Adaptive-D-Category",
    "Adaptive-D-highlighted_evidence",
    "Maladaptive-A-Category",
    "Maladaptive-A-highlighted_evidence",
    "Maladaptive-B-O-Category",
    "Maladaptive-B-O-highlighted_evidence",
    "Maladaptive-B-S-Category",
    "Maladaptive-B-S-highlighted_evidence",
    "Maladaptive-C-O-Category",
    "Maladaptive-C-O-highlighted_evidence",
    "Maladaptive-C-S-Category",
    "Maladaptive-C-S-highlighted_evidence",
    "Maladaptive-D-Category",
    "Maladaptive-D-highlighted_evidence",
    "post",
]

TEST_COLUMNS = [
    "timeline_id",
    "post_index",
    "post_id",
    "date",
    "date_number",
    "post",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def date_to_number(date_str: str) -> int:
    """
    Convert date string "DD-MM-YYYY, HH:MM:SS" to a numeric value.
    Here we use the ordinal day (integer number of days) so that
    a difference of < 8 corresponds to fewer than 8 days.
    """
    dt = datetime.strptime(date_str.strip(), DATE_FORMAT)
    return dt.toordinal()


def parse_state(state_dict: dict, prefix: str) -> dict:
    """
    Flatten one evidence state (adaptive or maladaptive) into columns.

    Output keys:
      <prefix>-Present
      <prefix>-<KEY>-Category
      <prefix>-<KEY>-highlighted_evidence
      for each KEY in EVIDENCE_KEYS.
    """
    result = {}

    # Presence: if missing, set to 1 as required
    presence = state_dict.get("Presence")
    if presence is None:
        presence = 1
    result[f"{prefix}-Present"] = presence

    for key in EVIDENCE_KEYS:
        if key in state_dict:
            entry = state_dict[key] or {}
            result[f"{prefix}-{key}-Category"] = entry.get("Category", "")
            result[f"{prefix}-{key}-highlighted_evidence"] = entry.get(
                "highlighted_evidence", ""
            )
        else:
            result[f"{prefix}-{key}-Category"] = ""
            result[f"{prefix}-{key}-highlighted_evidence"] = ""

    return result


def resolve_wellbeing(posts: list) -> list:
    """
    Fill null Well-being values within a timeline.

    Rule:
      - For each post with Well-being == None:
          If the *previous* post in the same timeline has a non-null
          Well-being AND |date_number difference| < 8,
          then copy that previous Well-being value.
          Otherwise set Well-being = 0.
    """
    if not posts:
        return posts

    # Work on a copy
    resolved = [dict(p) for p in posts]

    for i, post in enumerate(resolved):
        if post.get("Well-being") is None:
            if i > 0:
                prev = resolved[i - 1]
                prev_wb = prev.get("Well-being")
                if prev_wb is not None:
                    diff = abs(post["date_number"] - prev["date_number"])
                    if diff < 8:
                        resolved[i]["Well-being"] = prev_wb
                        continue
            # If no suitable previous, set to 0
            resolved[i]["Well-being"] = 0

    return resolved


# ── Training data processing ───────────────────────────────────────────────────
def process_train_file(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    timeline_id = data["timeline_id"]
    raw_posts = data["posts"]

    # First add date_number to each post
    posts = []
    for p in raw_posts:
        post_copy = dict(p)
        post_copy["date_number"] = date_to_number(post_copy["date"])
        posts.append(post_copy)

    # Resolve Well-being nulls according to rule
    posts = resolve_wellbeing(posts)

    rows = []
    for post in posts:
        evidence = post.get("evidence", {})
        adaptive_raw = evidence.get("adaptive-state", {}) or {}
        maladaptive_raw = evidence.get("maladaptive-state", {}) or {}

        adaptive_status = "no" if not adaptive_raw else "yes"
        maladaptive_status = "no" if not maladaptive_raw else "yes"

        adaptive_cols = parse_state(adaptive_raw, "Adaptive")
        maladaptive_cols = parse_state(maladaptive_raw, "Maladaptive")

        row = {
            "timeline_id": timeline_id,
            "post_id": post["post_id"],
            "date": post["date"],
            "date_number": post["date_number"],
            "post_index": post["post_index"],
            "Switch": post.get("Switch", ""),
            "Escalation": post.get("Escalation", ""),
            "Well-being": post.get("Well-being", 0),
            "Adaptive-status": adaptive_status,
            "Maladaptive-status": maladaptive_status,
            "Adaptive-Present": adaptive_cols["Adaptive-Present"],
            "Maladaptive-Present": maladaptive_cols["Maladaptive-Present"],
            "Adaptive-A-Category": adaptive_cols["Adaptive-A-Category"],
            "Adaptive-A-highlighted_evidence": adaptive_cols[
                "Adaptive-A-highlighted_evidence"
            ],
            "Adaptive-B-O-Category": adaptive_cols["Adaptive-B-O-Category"],
            "Adaptive-B-O-highlighted_evidence": adaptive_cols[
                "Adaptive-B-O-highlighted_evidence"
            ],
            "Adaptive-B-S-Category": adaptive_cols["Adaptive-B-S-Category"],
            "Adaptive-B-S-highlighted_evidence": adaptive_cols[
                "Adaptive-B-S-highlighted_evidence"
            ],
            "Adaptive-C-O-Category": adaptive_cols["Adaptive-C-O-Category"],
            "Adaptive-C-O-highlighted_evidence": adaptive_cols[
                "Adaptive-C-O-highlighted_evidence"
            ],
            "Adaptive-C-S-Category": adaptive_cols["Adaptive-C-S-Category"],
            "Adaptive-C-S-highlighted_evidence": adaptive_cols[
                "Adaptive-C-S-highlighted_evidence"
            ],
            "Adaptive-D-Category": adaptive_cols["Adaptive-D-Category"],
            "Adaptive-D-highlighted_evidence": adaptive_cols[
                "Adaptive-D-highlighted_evidence"
            ],
            "Maladaptive-A-Category": maladaptive_cols["Maladaptive-A-Category"],
            "Maladaptive-A-highlighted_evidence": maladaptive_cols[
                "Maladaptive-A-highlighted_evidence"
            ],
            "Maladaptive-B-O-Category": maladaptive_cols["Maladaptive-B-O-Category"],
            "Maladaptive-B-O-highlighted_evidence": maladaptive_cols[
                "Maladaptive-B-O-highlighted_evidence"
            ],
            "Maladaptive-B-S-Category": maladaptive_cols["Maladaptive-B-S-Category"],
            "Maladaptive-B-S-highlighted_evidence": maladaptive_cols[
                "Maladaptive-B-S-highlighted_evidence"
            ],
            "Maladaptive-C-O-Category": maladaptive_cols["Maladaptive-C-O-Category"],
            "Maladaptive-C-O-highlighted_evidence": maladaptive_cols[
                "Maladaptive-C-O-highlighted_evidence"
            ],
            "Maladaptive-C-S-Category": maladaptive_cols["Maladaptive-C-S-Category"],
            "Maladaptive-C-S-highlighted_evidence": maladaptive_cols[
                "Maladaptive-C-S-highlighted_evidence"
            ],
            "Maladaptive-D-Category": maladaptive_cols["Maladaptive-D-Category"],
            "Maladaptive-D-highlighted_evidence": maladaptive_cols[
                "Maladaptive-D-highlighted_evidence"
            ],
            "post": post["post"],
        }

        rows.append(row)

    return rows


def build_train_csv(train_dir: str, output_path: str) -> None:
    all_rows = []

    if not os.path.isdir(train_dir):
        print(f"Train directory '{train_dir}' does not exist.")
        return

    for filename in sorted(os.listdir(train_dir)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(train_dir, filename)
        all_rows.extend(process_train_file(filepath))

    if not all_rows:
        print("No training data found.")
        return

    df = pd.DataFrame(all_rows, columns=TRAIN_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved training CSV to {output_path}")


# ── Test data processing ───────────────────────────────────────────────────────
def process_test_file(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    timeline_id = data["timeline_id"]
    posts = data["posts"]

    rows = []
    for post in posts:
        row = {
            "timeline_id": timeline_id,
            "post_index": post["post_index"],
            "post_id": post["post_id"],
            "date": post["date"],
            "date_number": date_to_number(post["date"]),
            "post": post["post"],
        }
        rows.append(row)

    return rows


def build_test_csv(test_dir: str, output_path: str) -> None:
    all_rows = []

    if not os.path.isdir(test_dir):
        print(f"Test directory '{test_dir}' does not exist.")
        return

    for filename in sorted(os.listdir(test_dir)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(test_dir, filename)
        all_rows.extend(process_test_file(filepath))

    if not all_rows:
        print("No test data found.")
        return

    df = pd.DataFrame(all_rows, columns=TEST_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved test CSV to {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_train_csv(TRAIN_DIR, TRAIN_CSV)
    build_test_csv(TEST_DIR, TEST_CSV)
