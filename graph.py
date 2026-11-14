"""LangGraph workflow: shared state, routing rules, and a run() helper."""
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import config
from agents import make_nodes, pitch_text
from memory import TeamMemory
from trace_logger import Tracer


class PlannerState(TypedDict, total=False):
    domain: str
    research: dict
    funding: dict
    pitch: dict
    research_requests: list[str]  # Funding Advisor -> Research Agent
    review_issues: list[str]      # reviewers -> Pitch Coach
    research_round: int
    pitch_round: int
    approved: bool


def build_graph(llm, memory, tracer):
    nodes = make_nodes(llm, memory, tracer)

    def after_funding(state: PlannerState) -> str:
        wants_more = bool(state.get("research_requests"))
        if wants_more and state["research_round"] < config.MAX_RESEARCH_ROUNDS:
            tracer.log("route", "Orchestrator",
                       "Funding Advisor needs more research; looping back",
                       to="Research Agent", context="research_requests",
                       items=state["research_requests"])
            return "research"
        tracer.log("route", "Orchestrator", "Research is sufficient; moving to the pitch",
                   to="Pitch Coach", context="research report + funding plan")
        return "pitch"

    def after_review(state: PlannerState) -> str:
        if not state["approved"] and state["pitch_round"] < config.MAX_PITCH_ROUNDS:
            tracer.log("route", "Orchestrator", "Peers requested changes; looping back",
                       to="Pitch Coach", context="review_issues",
                       items=state["review_issues"])
            return "pitch"
        reason = "approved by both peers" if state["approved"] else "revision limit reached"
        tracer.log("route", "Orchestrator", f"Finishing: {reason}")
        return END

    builder = StateGraph(PlannerState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_edge(START, "research")
    builder.add_edge("research", "funding")
    builder.add_conditional_edges("funding", after_funding, ["research", "pitch"])
    builder.add_edge("pitch", "review")
    builder.add_conditional_edges("review", after_review, ["pitch", END])
    return builder.compile()


def run(domain: str, on_step=None, llm=None, embeddings=None):
    """Run the whole workflow. Returns (final_state, tracer, run_dir)."""
    slug = re.sub(r"[^a-z0-9]+", "-", domain.lower()).strip("-") or "startup"
    run_dir = Path("outputs") / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
    tracer = Tracer(run_dir)
    memory = TeamMemory(embeddings or config.get_embeddings(), tracer)
    graph = build_graph(llm or config.get_llm(), memory, tracer)

    try:  # save a diagram of the workflow next to the logs
        (run_dir / "graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    except Exception:
        pass

    state: PlannerState = {"domain": domain}
    for update in graph.stream({"domain": domain}, stream_mode="updates"):
        for node_name, changes in update.items():
            state.update(changes or {})
            if on_step:
                on_step(node_name, changes or {})

    (run_dir / "pitch_deck.md").write_text(pitch_text(state["pitch"]), encoding="utf-8")
    tracer.save_narrative()
    return state, tracer, run_dir
