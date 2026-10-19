"""Streamlit UI: a thin client of the FastAPI service (all logic lives in the API).

Run (with the API already running):
    streamlit run app/ui.py
    # MEDDOC_API_URL overrides the API address (default http://127.0.0.1:8000)
"""

import html
import os

import requests
import streamlit as st

# 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 ::1 first and costs ~2 s per request
# against a server bound to IPv4 (measured: 2,035 ms vs 16 ms for GET /health).
API_URL = os.environ.get("MEDDOC_API_URL", "http://127.0.0.1:8000").rstrip("/")
EXAMPLES = [
    "At what blood pressure should drug treatment for hypertension start?",
    "How much physical activity should older adults do each week?",
    "What should clinicians do before increasing opioid dosage to 90 MME per day?",
    "How long should hands be rubbed with an alcohol-based hand rub?",
    "Which antibiotic should be used to treat MRSA pneumonia?",  # out of scope: should be refused
]
REFUSAL_REASONS = {
    "unsafe_input": "The question looks like an attempt to override the assistant's instructions, so it was not processed.",
    "low_retrieval_confidence": "No sufficiently relevant passage was found in the indexed guidelines.",
    "llm_insufficient_context": "Relevant passages were found, but they do not contain the answer.",
    "no_valid_citations": "The draft answer could not be tied to any source, so it was withheld.",
}


def render_highlighted(text: str, spans: list[list[int]]) -> str:
    """Escape the chunk text and wrap highlight spans in <mark>."""
    out, pos = [], 0
    for start, end in sorted(spans):
        out.append(html.escape(text[pos:start]))
        out.append(f"<mark>{html.escape(text[start:end])}</mark>")
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def pages(s: dict) -> str:
    return f"p.{s['page_start']}" if s["page_start"] == s["page_end"] else f"pp.{s['page_start']}–{s['page_end']}"


st.set_page_config(page_title="MedDoc RAG", page_icon="🩺", layout="wide")
st.title("MedDoc RAG")
st.caption("Answers grounded in 8 public WHO, CDC and NIH guidelines, with page-level citations. "
           "Educational use only; not medical advice.")

with st.sidebar:
    st.subheader("Try an example")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state["question"] = q
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success("API connected")
        st.caption(f"Retriever: `{h['retriever']}` · chunks: `{h['chunk_strategy']}` ({h['indexed_chunks']})  \n"
                   f"Generator: `{h['generator']}` · refusal threshold: {h['refusal_threshold']}")
    except requests.RequestException:
        st.error(f"API not reachable at {API_URL}. Start it with `uvicorn app.api:app`.")

question = st.text_input("Your question", key="question", max_chars=500,
                         placeholder="e.g. What is the daily sodium limit for adults?")

if st.button("Ask", type="primary", disabled=not question or len(question.strip()) < 3):
    with st.spinner("Retrieving and generating..."):
        try:
            r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            st.stop()
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
        st.stop()
    data = r.json()

    if data.get("safety_notice"):
        st.error(f"**Urgent:** {data['safety_notice']}")
    conf = data["confidence"]
    if data["refused"]:
        st.warning(f"**{data['answer']}**  \n{REFUSAL_REASONS.get(data['refusal_reason'], '')}")
    else:
        st.markdown("### Answer")
        st.markdown(data["answer"])
        st.markdown("**Sources:** " + " · ".join(f"`[{c['number']}]` {c['label']}" for c in data["citations"]))
        if data["invalid_citations"]:
            st.caption(f"Removed citations to sources that were never provided: {data['invalid_citations']}")
        if data.get("removed_sentences"):
            st.caption(f"Removed {len(data['removed_sentences'])} sentence(s) containing links (output safety policy).")

    cols = st.columns(4)
    cols[0].metric("Retrieval confidence", "–" if conf is None else f"{conf:.2f}")
    lat = data["latency_ms"]
    cols[1].metric("Retrieval", f"{lat.get('retrieval', 0):.0f} ms")
    cols[2].metric("Generation", f"{lat.get('generation', 0):.0f} ms" if lat.get("generation") else "cached / none")
    cols[3].metric("Total", f"{lat.get('server_total', lat.get('total', 0)):.0f} ms")
    if conf is not None:
        st.progress(min(max(conf, 0.0), 1.0), text="Top reranker score (topical relevance, not a probability of correctness)")

    st.markdown("### Retrieved passages")
    for s in data["sources"]:
        badge = "✅ cited" if s["cited"] else "not cited"
        title = f"[{s['number']}] {s['doc_title']}, {pages(s)} · score {s['score']:.3f} · {badge}"
        with st.expander(title, expanded=s["cited"]):
            if s["section"]:
                st.caption(f"Section: {s['section']}")
            st.markdown(f"<div style='line-height:1.6'>{render_highlighted(s['text'], s['highlights'])}</div>",
                        unsafe_allow_html=True)
            if s["highlights"]:
                st.caption("Highlighted: sentences sharing the most words with the answer (approximate).")

    st.info(data["disclaimer"])
