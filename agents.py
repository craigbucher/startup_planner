"""Agent roles: output schemas, prompts, and the node functions LangGraph runs."""
from pydantic import BaseModel, Field

# ---------- Output schemas (each agent must explain its reasoning) ----------

# Defines a class for the research report output schema:
class ResearchReport(BaseModel):
    reasoning: str = Field(description="2-3 sentences on how you chose these insights")
    market_overview: str
    trends: list[str] = Field(description="4-6 current market trends")
    customer_segments: list[str]
    competitors: list[str]
    risks: list[str]

# Defines a class for individual funding options within a funding plan::
class FundingOption(BaseModel):
    name: str
    kind: str = Field(description="grant, accelerator, angel, venture capital, debt or other")
    stage: str
    typical_amount: str
    why_it_fits: str = Field(description="Tie this to a specific research finding")

# Defines a class for the overall funding plan, which includes multiple funding options:
class FundingPlan(BaseModel):
    reasoning: str = Field(description="2-3 sentences on how research shaped this plan")
    options: list[FundingOption] = Field(description="4-6 funding options")
    recommended_strategy: str
    research_requests: list[str] = Field(
        description="Specific gaps the Research Agent should fill. Empty if none."
    )

# Defines a class for an individual slide in a pitch deck:
class Slide(BaseModel):
    title: str
    key_points: list[str]
    based_on: str = Field(description="'research', 'funding' or 'both'")

# Defines a class for the overall pitch deck, which includes multiple slides:
class PitchDeck(BaseModel):
    reasoning: str = Field(description="2-3 sentences on the story arc you chose and why")
    startup_concept: str = Field(description="One-line concept the deck pitches")
    slides: list[Slide] = Field(description="10-12 slides")


# Defines a class for the review of a pitch deck:
class Review(BaseModel):
    reasoning: str
    approved: bool
    issues: list[str] = Field(description="Concrete fixes. Empty if approved.")

# ---------- Role definitions (scoped so they do not overlap) ----------
# Define the role for the Research Agent (market research):
RESEARCH_ROLE = (
    "You are the Research Agent in a startup accelerator. Your only job is market "
    "intelligence for a startup domain: market overview, trends, customer segments, "
    "competitors and risks. Do NOT recommend funding sources and do NOT write pitch "
    "content. Be specific; if you are unsure of a figure, say it is an estimate."
)
# Define the role for the Funding Advisor (funding recommendations):
FUNDING_ROLE = (
    "You are the Funding Advisor in a startup accelerator. Your only job is to "
    "recommend grants and funding programs that fit the research you are given. "
    "Do NOT redo market research and do NOT write pitch content. If the research "
    "lacks something you need (for example regulation, geography or capital "
    "intensity), list it in research_requests so the Research Agent can fill the gap."
)
# Define the role for the Pitch Coach (pitch deck creation):
PITCH_ROLE = (
    "You are the Pitch Coach in a startup accelerator. Your only job is to turn the "
    "Research Agent's insights and the Funding Advisor's plan into a compelling "
    "pitch deck outline. Do NOT invent market facts or funding sources that your "
    "peers did not provide. Include a funding-ask slide grounded in the funding plan."
)

# Helper function to format a list of items as bullet points:
def _bullets(items) -> str:
    return "\n".join(f"- {i}" for i in items) or "- (none)"

# Helper function to format the research report as text:
def research_text(r: dict) -> str:
    return (
        f"Overview: {r['market_overview']}\nTrends:\n{_bullets(r['trends'])}\n"
        f"Customer segments:\n{_bullets(r['customer_segments'])}\n"
        f"Competitors:\n{_bullets(r['competitors'])}\nRisks:\n{_bullets(r['risks'])}"
    )


# Helper function to format the funding plan as text:
def funding_text(f: dict) -> str:
    opts = [
        f"{o['name']} ({o['kind']}, {o['stage']}, {o['typical_amount']}): {o['why_it_fits']}"
        for o in f["options"]
    ]
    return f"Options:\n{_bullets(opts)}\nStrategy: {f['recommended_strategy']}"


# Helper function to format the pitch deck as text:
def pitch_text(p: dict) -> str:
    # Define a list to hold the formatted lines of the pitch deck:
    lines = [f"# Pitch deck outline: {p['startup_concept']}", ""]
    # Iterate over the slides and format each one:
    for n, s in enumerate(p["slides"], 1):
        # Add the slide title and the basis for the slide:
        lines.append(f"## Slide {n}: {s['title']}  _(based on {s['based_on']})_")
        # Add the key points for the slide:
        lines.extend(f"- {k}" for k in s["key_points"])
        lines.append("")
    # Join all the formatted lines into a single string and return it:
    return "\n".join(lines)

# Function to build the graph nodes:
def make_nodes(llm, memory, tracer) -> dict:
    """Build the four graph nodes. Each returns only the state keys it changes."""
    # Helper function to ask the LLM for structured output based on a schema, role, and task:
    def ask(schema, role: str, task: str) -> dict:
        # Log the task being sent to the LLM for debugging purposes:
        result = llm.with_structured_output(schema).invoke(
            [("system", role), ("human", task)]
        )
        # Return a dumped version of the result as a dictionary:
        return result.model_dump()

    # Node for conducting market research:
    def research_node(state: dict) -> dict:
        # Determine the current research round:
        round_no = state.get("research_round", 0) + 1
        # Get any previous research requests from the state:
        requests = state.get("research_requests", [])
        # Construct the initial task for the LLM:
        task = f"Startup domain: {state['domain']}\nProduce a market research report."
        # If there are previous research requests, append them to the task for refinement:
        if requests:  # feedback loop 1: refine using the Funding Advisor's requests
            task += (
                "\n\nYour previous report:\n" + research_text(state["research"])
                + "\n\nThe Funding Advisor asked you to fill these gaps. Keep what was "
                "good and add the missing detail:\n" + _bullets(requests)
            )
        # Ask the LLM for the research report based on the constructed task:
        report = ask(ResearchReport, RESEARCH_ROLE, task)
        # Log the research report and write relevant information to memory:
        tracer.log(
            "agent_output", "Research Agent",
            f"Research report, round {round_no} ({len(report['trends'])} trends)",
            reasoning=report["reasoning"], round=round_no, output=report,
        )
        # Write the research findings to memory:
        memory.write("Research Agent", "trend", report["trends"])
        # Write the identified risks to memory:
        memory.write("Research Agent", "risk", report["risks"])
        # Write the market overview to memory:
        memory.write("Research Agent", "market", [report["market_overview"]])
        # Log the handoff to the Funding Advisor:
        tracer.log("handoff", "Research Agent", "Passed research report on",
                to="Funding Advisor", context="full research report in shared state")
        # Return the updated state with the new research report and round number:
        return {"research": report, "research_round": round_no, "research_requests": []}

    # Node for planning funding strategy:
    def funding_node(state: dict) -> dict:
        # Construct the task for the Funding Advisor based on the research report:
        task = (
            f"Startup domain: {state['domain']}\n\nResearch Agent's report:\n"
            + research_text(state["research"])
            + "\n\nRecommend funding options that fit these findings."
        )
        # Ask the LLM for the funding plan based on the constructed task:
        plan = ask(FundingPlan, FUNDING_ROLE, task)
        # Log the funding plan to the tracer log:
        tracer.log(
            "agent_output", "Funding Advisor",
            f"Funding plan with {len(plan['options'])} options, "
            f"{len(plan['research_requests'])} research request(s)",
            reasoning=plan["reasoning"], output=plan,
        )
        # Write the funding options to memory:
        memory.write(
            "Funding Advisor", "funding option",
            [f"{o['name']}: {o['why_it_fits']}" for o in plan["options"]],
        )
        # Return the updated state with the new funding plan and research requests:
        return {"funding": plan, "research_requests": plan["research_requests"]}
    # Node for creating the pitch deck:
    def pitch_node(state: dict) -> dict:
        # Get the current pitch round number and increment it:
        round_no = state.get("pitch_round", 0) + 1
        # Recall relevant information from memory for the pitch:
        recalled = memory.recall(
            "Pitch Coach", f"strongest market trends and funding fit for {state['domain']}"
        )
        task = (
            f"Startup domain: {state['domain']}\n\nResearch:\n"
            + research_text(state["research"])
            + "\n\nFunding plan:\n" + funding_text(state["funding"])
            + "\n\nMost relevant items from team memory:\n" + _bullets(recalled)
        )
        # Check if there are any review issues from previous rounds:
        issues = state.get("review_issues", [])
        # If there are review issues, append them to the task for revision:
        if issues:  # feedback loop 2: revise using peer review
            task += (
                "\n\nYour previous draft:\n" + pitch_text(state["pitch"])
                + "\n\nYour peers reviewed it. Fix every issue:\n" + _bullets(issues)
            )
        # Add the instruction to write the pitch deck outline to the task:
        task += "\n\nWrite the pitch deck outline."
        # Ask the LLM to create the pitch deck based on the task:
        deck = ask(PitchDeck, PITCH_ROLE, task)
        # Log the pitch deck creation and handoff for peer review:
        tracer.log(
            "agent_output", "Pitch Coach",
            f"Pitch outline draft {round_no} ({len(deck['slides'])} slides)",
            reasoning=deck["reasoning"], round=round_no, output=deck,
        )
        # Write the pitch deck to the tracer log for tracking:
        tracer.log("handoff", "Pitch Coach", f"Sent draft {round_no} for peer review",
                to="Research Agent + Funding Advisor", context="pitch draft")
        # Return the updated state with the new pitch and round number:
        return {"pitch": deck, "pitch_round": round_no}

    # Review the pitch deck based on feedback from peers:
    def review_node(state: dict) -> dict:
        # Extract the current pitch draft for review:
        draft = pitch_text(state["pitch"])
        # Define the reviewers and their focus areas:
        reviewers = [
            ("Research Agent", RESEARCH_ROLE, research_text(state["research"]),
            "market claims that contradict, exaggerate or omit your research"),
            ("Funding Advisor", FUNDING_ROLE, funding_text(state["funding"]),
            "a funding ask or use of funds that does not match your funding plan"),
        ]
        issues, approvals = [], []
        # Iterate over each reviewer and collect their feedback:
        for name, role, own_work, focus in reviewers:
            # Ask the reviewer to evaluate the pitch draft based on their expertise and focus area:
            review = ask(
                Review, role,
                "You are now reviewing a peer's work, only within your own expertise.\n"
                f"Your findings:\n{own_work}\n\nPitch Coach's draft:\n{draft}\n\n"
                f"Check for {focus}. Approve only if there is nothing important to fix.",
            )
            # Log the reviewer's feedback for tracking:
            tracer.log(
                "agent_output", name,
                "Approved the pitch draft" if review["approved"]
                else f"Requested {len(review['issues'])} change(s) to the pitch draft",
                reasoning=review["reasoning"], items=review["issues"], output=review,
            )
            # Record whether the reviewer approved the pitch draft:
            approvals.append(review["approved"])
            # Append the reviewer's issues to the overall list of issues:
            issues.extend(f"{name}: {i}" for i in review["issues"])
        # Return the overall approval status and collected review issues:
        return {"approved": all(approvals), "review_issues": issues}

    # Return the dictionary of agent nodes for the startup planning process:
    return {
        "research": research_node,
        "funding": funding_node,
        "pitch": pitch_node,
        "review": review_node,
    }
