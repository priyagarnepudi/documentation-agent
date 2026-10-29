"""
Streamlit app for exploring LLM judge results.

Browse judged eval entries, filter by score across all three checks
(answer correctness, instruction following, trajectory optimality),
and inspect detailed reasoning for each entry.

Usage:
    uv run streamlit run evals/synthetic/app.py
"""

import streamlit as st
import json
import os
import glob

from doc_agent import DEFAULT_INSTRUCTIONS
from evals.utils import GITHUB_BASE

CHECK_KEYS = [
    ("Answer Correctness", "judge_answer_correctness"),
    ("Instruction Following", "judge_instruction_following"),
    ("Trajectory Optimality", "judge_trajectory"),
]


def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)


def score_icon(score):
    return "✅" if score == "good" else "❌"


def score_summary(entry):
    """Return a compact string of score icons for sidebar display."""
    parts = []
    for _, key in CHECK_KEYS:
        judge = entry.get(key, {})
        parts.append(score_icon(judge.get("score", "bad")))
    return " ".join(parts)


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Judge Results Explorer")

# ── File selection ──────────────────────────────────────────────────────────
st.sidebar.header("📂 Data File")
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
json_files = sorted(glob.glob(os.path.join(data_dir, "*_judged.json")))

if not json_files:
    st.error("No *_judged.json files found in the evals directory. Run `run_judge_checks.py` first.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select judged file",
    json_files,
    format_func=lambda f: os.path.basename(f),
    key="file_selector",
)

# ── Load data ───────────────────────────────────────────────────────────────
if "loaded_file" not in st.session_state or st.session_state.loaded_file != selected_file:
    st.session_state.loaded_file = selected_file
    st.session_state.data = load_data(selected_file)
    st.session_state.current_index = 0

data = st.session_state.data

if not data:
    st.error(f"Could not load data from {selected_file}")
    st.stop()

# ── Filters ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

filter_correctness = st.sidebar.selectbox(
    "Answer Correctness",
    ["all", "good", "bad"],
    key="filter_correctness",
)
filter_instruction = st.sidebar.selectbox(
    "Instruction Following",
    ["all", "good", "bad"],
    key="filter_instruction",
)
filter_trajectory = st.sidebar.selectbox(
    "Trajectory Optimality",
    ["all", "good", "bad"],
    key="filter_trajectory",
)

# Apply filters
filters = {
    "judge_answer_correctness": filter_correctness,
    "judge_instruction_following": filter_instruction,
    "judge_trajectory": filter_trajectory,
}

filtered_data = []
for i, entry in enumerate(data):
    match = True
    for key, value in filters.items():
        if value != "all":
            if entry.get(key, {}).get("score", "bad") != value:
                match = False
                break
    if match:
        filtered_data.append((i, entry))

# ── Overview stats ──────────────────────────────────────────────────────────
st.sidebar.header("📊 Overview")
total = len(data)
for label, key in CHECK_KEYS:
    good = sum(1 for e in data if e.get(key, {}).get("score") == "good")
    st.sidebar.caption(f"{label}: **{good}/{total}** good ({good/total*100:.0f}%)")

st.sidebar.divider()
st.sidebar.caption(f"Showing **{len(filtered_data)}** / {total} entries")

# ── Entry selector ──────────────────────────────────────────────────────────
if not filtered_data:
    st.warning("No entries match the selected filters.")
    st.stop()

st.sidebar.header("🧭 Navigation")
selection = st.sidebar.selectbox(
    "Select Entry",
    range(len(filtered_data)),
    index=0,
    format_func=lambda i: (
        f"{score_summary(filtered_data[i][1])} "
        f"#{filtered_data[i][0]+1}: "
        f"{filtered_data[i][1]['input']['question'][:40]}..."
    ),
)

original_idx, entry = filtered_data[selection]
rag = entry.get("rag_response", {})
inp = entry.get("input", {})

# ── Title ───────────────────────────────────────────────────────────────────
st.title(f"Entry #{original_idx + 1} / {total}")

# Score badges
badge_cols = st.columns(3)
for col, (label, key) in zip(badge_cols, CHECK_KEYS):
    score = entry.get(key, {}).get("score", "bad")
    icon = score_icon(score)
    col.metric(label, f"{icon} {score}")

st.divider()

# ── Input data ──────────────────────────────────────────────────────────────
st.subheader("❓ Question & Reference")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**User Question**")
    st.chat_message("user").write(inp.get("question", ""))
with col2:
    st.markdown("**Reference Answer**")
    st.info(inp.get("reference_answer", "N/A"))

meta_cols = st.columns(4)
meta_cols[0].caption(f"**Type:** {inp.get('question_type', 'N/A')}")
meta_cols[1].caption(f"**File:** {inp.get('filename', 'N/A')}")
meta_cols[2].caption(f"**Lines:** {inp.get('line_number_start', '?')}–{inp.get('line_number_end', '?')}")
meta_cols[3].caption(f"**Agent cost:** ${entry.get('cost', 0):.4f}")

st.divider()

# ── Agent answer ────────────────────────────────────────────────────────────
st.subheader("🤖 Agent Answer")
with st.chat_message("assistant"):
    st.markdown(rag.get("answer", "No answer found."))

# Metadata row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Confidence", rag.get("confidence", 0))
    st.caption(rag.get("confidence_explanation", ""))
with col2:
    st.markdown("**Answer Type**")
    st.text(rag.get("answer_type", "N/A"))
    st.markdown("**Found Answer**")
    st.text(str(rag.get("found_answer", "N/A")))
with col3:
    st.markdown("**Self-Checks**")
    for check in rag.get("checks", []):
        icon = "✅" if check["passed"] else "❌"
        st.markdown(f"{icon} {check['rule']}")

# References
refs = rag.get("references", [])
if refs:
    st.markdown("**📚 References**")
    for ref in refs:
        file_path = ref["file_path"]
        url = GITHUB_BASE + file_path
        st.markdown(f"- 📄 [{file_path}]({url}): {ref['explanation']}")

# Follow-up questions
followups = rag.get("followup_questions", [])
if followups:
    with st.expander("Follow-up Questions"):
        for q in followups:
            st.markdown(f"- {q}")

st.divider()

# ── Tool calls (trajectory) ────────────────────────────────────────────────
st.subheader("🛠️ Tool Calls (Trajectory)")
tools = entry.get("tools", [])
if tools:
    for i, tool in enumerate(tools):
        st.markdown(f"**{i+1}. {tool['name']}**")
        st.code(tool["args"], language="json")
else:
    st.caption("No tool calls recorded.")

st.divider()

# ── Judge results ───────────────────────────────────────────────────────────
st.subheader("⚖️ Judge Evaluations")

tabs = st.tabs([label for label, _ in CHECK_KEYS])

for tab, (label, key) in zip(tabs, CHECK_KEYS):
    with tab:
        judge = entry.get(key, {})
        score = judge.get("score", "bad")
        reasoning = judge.get("reasoning", "No reasoning available.")

        st.markdown(f"**Score:** {score_icon(score)} **{score}**")
        st.markdown("**Reasoning:**")
        st.markdown(reasoning)

        # Trajectory has an extra suggestion field
        suggestion = judge.get("suggestion")
        if suggestion and suggestion != "none":
            st.markdown("**💡 Suggestion:**")
            st.info(suggestion)

st.divider()

# ── Agent instructions (collapsed) ──────────────────────────────────────────
with st.expander("📋 Agent Instructions (System Prompt)"):
    st.markdown(DEFAULT_INSTRUCTIONS)
