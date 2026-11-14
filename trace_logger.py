"""Structured local logging: one JSON line per event, plus a readable narrative."""
import json
from datetime import datetime, timezone
from pathlib import Path

# The Tracer class provides structured logging for agent events, storing 
# them as JSON lines and generating a human-readable narrative:
class Tracer:
    def __init__(self, run_dir: Path):
        # Ensure the run directory exists and initialize the path for the trace file:
        self.run_dir = Path(run_dir)
        # Store the path to the trace file within the run directory:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Initialize the trace file path:
        self.path = self.run_dir / "trace.jsonl"
        # Initialize the list of events to keep track of logged records:
        self.events: list[dict] = []

    # Log an event with the given details, storing it in the trace and returning the record:
    def log(self, event: str, agent: str, summary: str, **data) -> dict:
        """event is one of: agent_output, handoff, memory_write, memory_read, route."""
        record = {
            # Record the step number:
            "step": len(self.events) + 1,
            # Record the current UTC time of the event:
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # Record the type of event:
            "event": event,
            # Record the agent responsible for the event:
            "agent": agent,
            # Record the summary of the event:
            "summary": summary,
            # Include any additional data provided for the event:
            **data,
        }
        # Append the record to the list of events and write it to the trace file:
        self.events.append(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        # Return the record for potential further use:
        return record

    # Generate a human-readable narrative of the logged events:
    def narrative(self) -> str:
        """Narrate, step by step, how the agents arrived at the final output."""
        # Start building the narrative with a header:
        # Initialize the list of lines that will make up the narrative:
        lines = ["# Reasoning trace", ""]
        # Iterate over each logged event to build the narrative:
        for e in self.events:
            # Add a line describing the current step, agent, event type, and summary:
            lines.append(f"**Step {e['step']} · {e['agent']} · {e['event']}**: {e['summary']}")
            # Include reasoning if available:
            if e.get("reasoning"):
                lines.append(f"> Reasoning: {e['reasoning']}")
            # Include handoff information if available:
            if e.get("to"):
                lines.append(f"> Handed to: {e['to']} ({e.get('context', '')})")
            # Include items if available:
            if e.get("items"):
                lines.extend(f"> - {item}" for item in e["items"])
            # Add an empty line for readability between steps:
            lines.append("")
        # Return the complete narrative as a single string:
        return "\n".join(lines)

    # Save the generated narrative to a markdown file and return the path:
    def save_narrative(self) -> Path:
        # Determine the output file path within the run directory:
        out = self.run_dir / "trace.md"
        # Write the narrative to the file using UTF-8 encoding:
        out.write_text(self.narrative(), encoding="utf-8")
        # Return the path to the saved narrative file:
        return out
