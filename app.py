"""Streamlit interface:  streamlit run app.py"""
import json

import streamlit as st

from agents import funding_text, pitch_text, research_text
from graph import run

st.set_page_config(page_title="Startup Accelerator Planner", layout="wide")
st.title("Multi-Agent Workflow Planner")
st.caption("Research Agent → Funding Advisor → Pitch Coach, with peer feedback loops")

domain = st.text_input("Startup domain or industry", placeholder="e.g. fintech, healthtech")

if st.button("Generate pitch deck outline", type="primary") and domain.strip():
    progress = st.status("Agents are working...", expanded=True)

    def on_step(node, changes):
        if node == "research":
            progress.write(f"Research Agent: report round {changes['research_round']}")
        elif node == "funding":
            n = len(changes["research_requests"])
            progress.write(f"Funding Advisor: plan ready, {n} research request(s)")
        elif node == "pitch":
            progress.write(f"Pitch Coach: draft {changes['pitch_round']}")
        elif node == "review":
            verdict = "approved" if changes["approved"] else "changes requested"
            progress.write(f"Peer review: {verdict}")

    state, tracer, run_dir = run(domain.strip(), on_step=on_step)
    progress.update(label="Done", state="complete", expanded=False)

    deck_tab, work_tab, trace_tab, raw_tab = st.tabs(
        ["Pitch deck outline", "Agent contributions", "Reasoning trace", "Raw log"]
    )
    with deck_tab:
        deck_md = pitch_text(state["pitch"])
        st.markdown(deck_md)
        st.download_button("Download outline (.md)", deck_md, file_name="pitch_deck.md")
    with work_tab:
        left, right = st.columns(2)
        left.subheader("Research Agent")
        left.text(research_text(state["research"]))
        right.subheader("Funding Advisor")
        right.text(funding_text(state["funding"]))
    with trace_tab:
        st.markdown(tracer.narrative())
    with raw_tab:
        st.caption(f"Saved to {run_dir}/trace.jsonl")
        for event in tracer.events:
            with st.expander(f"{event['step']}. {event['agent']}: {event['summary']}"):
                st.code(json.dumps(event, indent=2, default=str), language="json")
