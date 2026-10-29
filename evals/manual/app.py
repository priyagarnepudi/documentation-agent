import streamlit as st
import streamlit.components.v1
import json
import os
import glob

from evals.manual.judge import DEFAULT_INSTRUCTIONS
from evals.utils import GITHUB_BASE


def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        return json.load(f)


def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def get_results_path(input_path):
    """Given an input JSON path, return the corresponding _results.json path."""
    base, ext = os.path.splitext(input_path)
    return base + "_results" + ext


def find_next_unlabelled(data, start_index):
    """Find the next unlabelled item starting from start_index+1, wrapping around."""
    n = len(data)
    for offset in range(1, n):
        i = (start_index + offset) % n
        if data[i].get('label') is None:
            return i
    return None


st.set_page_config(layout="wide", page_title="Agent Eval Labeling")

# ── Scroll anchor ──────────────────────────────────────────────────────────
st.markdown('<div id="scroll-top-anchor"></div>', unsafe_allow_html=True)

if st.session_state.get('_scroll_top', False):
    st.session_state._scroll_top = False
    js = """
        <script>
        function scrollToTop(attempts) {
            const doc = window.parent.document;
            const anchor = doc.getElementById('scroll-top-anchor');
            if (anchor) {
                anchor.scrollIntoView({behavior: 'instant', block: 'start'});
                return;
            }
            if (attempts > 0) {
                setTimeout(function() { scrollToTop(attempts - 1); }, 50);
            }
        }
        // Try multiple times    uv run streamlit run evals/manual/app.py be rendering
        setTimeout(function() { scrollToTop(10); }, 100);
        </script>
    """
    st.components.v1.html(js, height=0)

# ── File Selection ──────────────────────────────────────────────────────────
st.sidebar.header("📂 File Selection")
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
input_files = [f for f in json_files if not f.endswith("_results.json")]

if not input_files:
    st.error("No JSON files found in the evals directory.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select evaluation file",
    input_files,
    format_func=lambda f: os.path.basename(f),
    key="file_selector"
)

results_file = get_results_path(selected_file)

# ── Load Data ───────────────────────────────────────────────────────────────
if 'selected_file' not in st.session_state or st.session_state.selected_file != selected_file:
    st.session_state.selected_file = selected_file
    if os.path.exists(results_file):
        st.session_state.data = load_data(results_file)
    else:
        st.session_state.data = load_data(selected_file)
        for item in st.session_state.data:
            if 'label' not in item:
                item['label'] = None
            if 'comments' not in item:
                item['comments'] = ""
    first_unlabelled = next(
        (i for i, item in enumerate(st.session_state.data) if item.get('label') is None), 0
    )
    st.session_state.current_index = first_unlabelled

if not st.session_state.data:
    st.error(f"Could not load data from {selected_file}")
    st.stop()

# ── Navigation ──────────────────────────────────────────────────────────────
st.sidebar.header("🧭 Navigation")
selection = st.sidebar.selectbox(
    "Select Run Result",
    range(len(st.session_state.data)),
    index=st.session_state.current_index,
    format_func=lambda i: f"{'✅' if st.session_state.data[i].get('label') == 'good' else '❌' if st.session_state.data[i].get('label') == 'bad' else '⬜'} Run {i+1}: {st.session_state.data[i]['input']['question'][:30]}..."
)

if selection != st.session_state.current_index:
    st.session_state.current_index = selection
    st.session_state._scroll_top = True
    st.rerun()

# Progress
labeled_count = sum(1 for item in st.session_state.data if item.get('label') is not None)
st.sidebar.progress(labeled_count / len(st.session_state.data))
st.sidebar.text(f"Labeled: {labeled_count} / {len(st.session_state.data)}")

# ── Completion banner ───────────────────────────────────────────────────────
if labeled_count == len(st.session_state.data):
    st.success("🎉 Everything is labeled! Great job!")
    st.divider()

item = st.session_state.data[st.session_state.current_index]
rag = item.get('rag_response', {})

st.title(f"Run {st.session_state.current_index + 1} / {len(st.session_state.data)}")

# ── User Question ───────────────────────────────────────────────────────────
st.subheader("❓ User Question")
st.chat_message("user").write(item['input']['question'])

# ── System Answer ───────────────────────────────────────────────────────────
st.subheader("📝 System Answer")
with st.chat_message("assistant"):
    st.markdown(rag.get('answer', 'No answer found.'))

    refs = rag.get('references', [])
    if refs:
        st.markdown("**📚 References**")
        for ref in refs:
            file_path = ref['file_path']
            url = GITHUB_BASE + file_path
            st.markdown(f"- 📄 [{file_path}]({url}): {ref['explanation']}")

# ── Metadata ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Confidence Score", rag.get('confidence', 0))
    st.caption(rag.get('confidence_explanation', ''))
with col2:
    st.markdown("**Answer Type**")
    st.text(rag.get('answer_type', 'N/A'))
with col3:
    st.markdown("**Checks**")
    checks = rag.get('checks', [])
    for check in checks:
        icon = "✅" if check['passed'] else "❌"
        st.markdown(f"{icon} {check['rule']}")

# ── Tool Calls & Instructions (collapsed) ──────────────────────────────────
st.subheader("🛠️ Tool Calls")
with st.expander("View Tool Calls"):
    for i, tool in enumerate(item.get('tools', [])):
        st.markdown(f"**Tool {i+1}: {tool['name']}**")
        st.code(tool['args'], language='json')

st.subheader("📋 Agent Instructions")
with st.expander("View System Instructions"):
    st.markdown(DEFAULT_INSTRUCTIONS)

st.divider()

# ── Labeling Decision (at the bottom) ───────────────────────────────────────
st.subheader("⚖️ Labeling Decision")
label_col1, label_col2 = st.columns([1, 2])

with label_col1:
    current_label = item.get('label')
    options = ["Not Selected", "Good", "Not Good"]
    default_idx = 0
    if current_label == 'good':
        default_idx = 1
    elif current_label == 'bad':
        default_idx = 2

    decision = st.radio(
        "Is this run good?",
        options,
        index=default_idx,
        key=f"radio_{st.session_state.current_index}"
    )

with label_col2:
    comments = st.text_area(
        "Comments",
        value=item.get('comments', ""),
        height=100,
        key=f"comments_{st.session_state.current_index}"
    )

if st.button("💾 Save Label & Next", type="primary"):
    if decision == "Good":
        item['label'] = 'good'
    elif decision == "Not Good":
        item['label'] = 'bad'
    else:
        item['label'] = None
    item['comments'] = comments

    save_data(st.session_state.data, results_file)

    next_idx = find_next_unlabelled(st.session_state.data, st.session_state.current_index)
    if next_idx is not None:
        st.session_state.current_index = next_idx
    # Flag scroll-to-top for next rerun
    st.session_state._scroll_top = True
    st.rerun()

# ── Sidebar Reset ───────────────────────────────────────────────────────────
st.sidebar.divider()
if st.sidebar.button("RESET LABELS (Caution)"):
    if st.sidebar.checkbox("Confirm Reset"):
        if os.path.exists(results_file):
            os.remove(results_file)
        for key in ['data', 'current_index', 'selected_file']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
