# Multi-Agent Workflow Planner for a Startup Accelerator

Given a startup domain, three AI agents collaborate to produce a pitch deck
outline, with a full trace of how they got there.

## Quick start

1. `python -m venv .venv` and activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add an OpenAI or Azure OpenAI key
4. `python main.py "fintech"` or `streamlit run app.py`

Outputs for each run are saved under `outputs/<timestamp>-<domain>/`.

## System architecture

- **Orchestration:** LangGraph `StateGraph` with four nodes and two
  conditional edges (`graph.py`).
- **Shared state:** one typed dictionary carries the domain, each agent's
  output, feedback lists and loop counters between nodes.
- **Vector memory:** agents write findings to an in-memory vector store;
  the Pitch Coach recalls the most relevant items (`memory.py`).
- **Models:** OpenAI or Azure OpenAI, switched by `LLM_PROVIDER` (`config.py`).
- **Tracing:** structured JSONL log plus a narrated trace (`trace_logger.py`);
  optional LangSmith.

Flow:

    domain -> Research Agent -> Funding Advisor -> Pitch Coach -> Peer review -> outline
                    ^                 |                ^               |
                    +- research gaps -+                +--- issues ----+

## Why LangGraph

TODO: write your justification in your own words (loops, explicit state,
fixed flow, built-in tracing). See Step 3 of the build guide.

## Agents

| Agent | Responsibility | Input | Output |
| --- | --- | --- | --- |
| Research Agent | Market trends, segments, competitors, risks | Domain, gap requests | `ResearchReport` |
| Funding Advisor | Grants and funding programs that fit the research | Research report | `FundingPlan` |
| Pitch Coach | Pitch deck outline | Research, funding plan, memory, reviews | `PitchDeck` |

Each prompt states what the agent must not do, so responsibilities do not
overlap. Every output includes a `reasoning` field that is written to the trace.

## Coordination and feedback loops

1. The Research Agent reports on the domain.
2. The Funding Advisor builds a plan. If the research has gaps, it sends
   `research_requests` back and the Research Agent refines its report.
3. The Pitch Coach drafts the outline from both outputs plus memory recall.
4. The Research Agent and Funding Advisor review the draft against their own
   findings. If either objects, the Pitch Coach revises.
5. Loop limits (`MAX_RESEARCH_ROUNDS`, `MAX_PITCH_ROUNDS`) guarantee the run ends.

## Logging and traceability

- `trace.jsonl`: every agent output, hand-off, memory read/write and routing
  decision, in order.
- `trace.md`: the same events narrated step by step.
- `graph.mmd`: Mermaid diagram of the workflow.
- Streamlit tabs show the trace and raw events interactively.

TODO: add a screenshot of the trace tab and a short excerpt of trace.md.

## Challenges and trade-offs

- **No live data.** The Research Agent uses the model's knowledge, so figures
  and named programs need verification. A web search tool would fix this at
  the cost of another API key and slower runs.
- **Bounded loops.** Capping rounds keeps cost predictable but can end a run
  before reviewers are fully satisfied.
- **Structured output.** Schemas make hand-offs reliable and loggable but
  constrain how freely agents can respond.
- **In-memory vector store.** Simple and dependency-free, but memory is lost
  between runs; a persistent store would let agents learn across founders.
- TODO: add the problems you actually hit and how you solved them.
