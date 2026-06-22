"""
Case Closed — Railtracks agent template.

Copy this file to your project, set DETECTIVE_TOKEN in .env, and run:

    python agent_template.py

The agent will investigate S001, call game actions as tools, and commit
a structured answer when it's confident.
"""

import yaml
import railtracks as rt
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel

from detective_client import (
    DetectiveClient,
    MoveObservation,
    SearchObservation,
    InterviewObservation,
    PresentObservation,
    GameError,
)

from dotenv import load_dotenv
load_dotenv()

_prompts = yaml.safe_load((Path(__file__).parent / "prompts.yaml").read_text(encoding="utf-8"))


rt.enable_logging()
# One client per run — tracks session, clues, and inventory automatically
client = DetectiveClient()


@dataclass
class InvestigationTracker:
    """Deterministic memory for facts the LLM should not have to remember."""

    all_locations: set[str] = field(default_factory=set)
    all_characters: set[str] = field(default_factory=set)
    current_location: str = ""
    visited: set[str] = field(default_factory=set)
    searched: set[str] = field(default_factory=set)
    interviewed: set[str] = field(default_factory=set)
    exits_by_location: dict[str, set[str]] = field(default_factory=dict)
    characters_by_location: dict[str, set[str]] = field(default_factory=dict)
    searchable_locations: set[str] = field(default_factory=set)
    clues: dict[str, str] = field(default_factory=dict)
    inventory: set[str] = field(default_factory=set)
    presentations: set[tuple[str, str]] = field(default_factory=set)

    def reset(self, session) -> None:
        self.all_locations = {location.id for location in session.briefing.locations}
        self.all_characters = {character.id for character in session.briefing.cast}
        self.current_location = ""
        self.visited.clear()
        self.searched.clear()
        self.interviewed.clear()
        self.exits_by_location.clear()
        self.characters_by_location.clear()
        self.searchable_locations.clear()
        self.clues.clear()
        self.inventory.clear()
        self.presentations.clear()
        self.record_location(session.observation)

    def record_location(self, observation) -> None:
        location_id = observation.location_id
        self.current_location = location_id
        self.visited.add(location_id)
        self.exits_by_location[location_id] = set(observation.exits)
        self.characters_by_location[location_id] = set(observation.characters_present)
        if observation.searchable:
            self.searchable_locations.add(location_id)

    def summary(self, actions_remaining: int) -> str:
        unvisited = sorted(self.all_locations - self.visited)
        unsearched = sorted(self.searchable_locations - self.searched)
        missing_characters = sorted(self.all_characters - self.interviewed)
        exits = sorted(self.exits_by_location.get(self.current_location, set()))
        map_lines = [
            f"  {location} -> {sorted(destinations)}"
            for location, destinations in sorted(self.exits_by_location.items())
        ]
        character_lines = [
            f"  {location} -> {sorted(characters)}"
            for location, characters in sorted(self.characters_by_location.items())
            if characters
        ]
        clue_lines = [
            f"  [{clue_id}] {text}" for clue_id, text in sorted(self.clues.items())
        ]
        presentation_lines = [
            f"  {character_id} <- {ref_id}"
            for character_id, ref_id in sorted(self.presentations)
        ]
        return (
            f"Current location  : {self.current_location}\n"
            f"Current exits     : {exits}\n"
            f"Actions remaining : {actions_remaining}\n"
            f"Visited           : {sorted(self.visited)}\n"
            f"Unvisited         : {unvisited or 'none'}\n"
            f"Searched          : {sorted(self.searched)}\n"
            f"Unsearched known  : {unsearched or 'none'}\n"
            f"Interviewed       : {sorted(self.interviewed)}\n"
            f"Not interviewed   : {missing_characters or 'none'}\n"
            f"Inventory         : {sorted(self.inventory) or 'none'}\n"
            f"Known map:\n{chr(10).join(map_lines) or '  none'}\n"
            f"Known characters:\n{chr(10).join(character_lines) or '  none'}\n"
            f"Collected clues:\n{chr(10).join(clue_lines) or '  none'}\n"
            f"Presentations tried:\n{chr(10).join(presentation_lines) or '  none'}"
        )


tracker = InvestigationTracker()


# ── Tools (game actions) ──────────────────────────────────────────────────────

@rt.function_node
def move_to(location_id: str) -> str:
    """Move to an adjacent location on the map.

    Args:
        location_id: Destination location ID (e.g. 'loc_office', 'loc_alley').
    Returns:
        Description of the new location, characters present, and available exits.
    """
    exits = tracker.exits_by_location.get(tracker.current_location, set())
    if exits and location_id not in exits:
        return (
            f"Move blocked before spending an action: '{location_id}' is not an exit from "
            f"{tracker.current_location}. Valid exits: {sorted(exits)}"
        )
    try:
        obs: MoveObservation = client.move(location_id)
    except GameError as exc:
        return f"Move rejected: {exc}"
    tracker.record_location(obs)
    chars = ", ".join(obs.characters_present) or "nobody"
    exits = ", ".join(obs.exits)
    return (
        f"Moved to {obs.name}.\n"
        f"{obs.description}\n"
        f"Characters here: {chars}\n"
        f"Exits: {exits}\n"
        f"Searchable: {obs.searchable} | Actions left: {obs.actions_remaining}"
    )


@rt.function_node
def search_location() -> str:
    """Search the current location for physical evidence and items.

    Returns:
        Each piece of evidence found, the clues it reveals, and any items
        added to your inventory.
    """
    if tracker.current_location not in tracker.searchable_locations:
        return "Search blocked: the current location is not searchable."
    if tracker.current_location in tracker.searched:
        return "Search skipped: this location was already searched."
    try:
        obs: SearchObservation = client.search()
    except GameError as exc:
        return f"Search rejected: {exc}"
    tracker.searched.add(tracker.current_location)
    for evidence in obs.found:
        if evidence.item_id:
            tracker.inventory.add(evidence.item_id)
        for clue in evidence.yields:
            tracker.clues[clue.clue_id] = clue.text
    if not obs.found:
        return f"Nothing found here. Actions left: {obs.actions_remaining}"
    lines = []
    for ev in obs.found:
        clue_texts = " | ".join(f"[{c.clue_id}] {c.text}" for c in ev.yields)
        item_note = f" → item '{ev.item_id}' added to inventory" if ev.item_id else ""
        lines.append(f"• {ev.name}{item_note}\n  Clues: {clue_texts}")
    lines.append(f"Actions left: {obs.actions_remaining}")
    return "\n".join(lines)


@rt.function_node
def interview_character(character_id: str) -> str:
    """Talk to a character at your current location to hear their testimony.

    Characters always give the same freely offered clues regardless of how many
    times you ask. Use present_to_character to unlock additional guarded clues.

    Args:
        character_id: Who to talk to (e.g. 'char_finn', 'char_silas').
    Returns:
        Everything the character says, with clue IDs.
    """
    characters_here = tracker.characters_by_location.get(tracker.current_location, set())
    if character_id not in characters_here:
        return (
            f"Interview blocked: '{character_id}' is not at {tracker.current_location}. "
            f"Characters here: {sorted(characters_here) or 'nobody'}"
        )
    if character_id in tracker.interviewed:
        return "Interview skipped: this character was already interviewed."
    try:
        obs: InterviewObservation = client.interview(character_id)
    except GameError as exc:
        return f"Interview rejected: {exc}"
    tracker.interviewed.add(character_id)
    for clue in obs.said:
        tracker.clues[clue.clue_id] = clue.text
    if not obs.said:
        return f"{obs.name} has nothing to say. Actions left: {obs.actions_remaining}"
    lines = [f"{obs.name} says:"]
    for clue in obs.said:
        lines.append(f"  [{clue.clue_id}] {clue.text}")
    lines.append(f"Actions left: {obs.actions_remaining}")
    return "\n".join(lines)


@rt.function_node
def present_to_character(character_id: str, ref_id: str) -> str:
    """Show an item or clue to a character to unlock their guarded testimony.

    Use an item_id (from inventory after searching) or a clue_id (already seen)
    as ref_id. The character will only respond if it matches their unlock condition.

    Args:
        character_id: Who to confront (e.g. 'char_mara', 'char_silas').
        ref_id: The item_id or clue_id to present (e.g. 'item_ledger', 'clue_07').
    Returns:
        Any new clues the character reveals, or a note that they were unpersuaded.
    """
    characters_here = tracker.characters_by_location.get(tracker.current_location, set())
    if character_id not in characters_here:
        return (
            f"Present blocked: '{character_id}' is not at {tracker.current_location}. "
            f"Characters here: {sorted(characters_here) or 'nobody'}"
        )
    if ref_id not in tracker.clues and ref_id not in tracker.inventory:
        return f"Present blocked: '{ref_id}' has not been collected."
    attempt = (character_id, ref_id)
    if attempt in tracker.presentations:
        return "Present skipped: this exact character/reference pair was already tried."
    try:
        obs: PresentObservation = client.present(character_id, ref_id=ref_id)
    except GameError as exc:
        return f"Present rejected: {exc}"
    tracker.presentations.add(attempt)
    for clue in obs.revealed:
        tracker.clues[clue.clue_id] = clue.text
    if not obs.accepted or not obs.revealed:
        return f"Character was not persuaded by '{ref_id}'. Actions left: {obs.actions_remaining}"
    lines = ["Testimony unlocked:"]
    for clue in obs.revealed:
        lines.append(f"  [{clue.clue_id}] {clue.text}")
    lines.append(f"Actions left: {obs.actions_remaining}")
    return "\n".join(lines)


@rt.function_node
def get_investigation_summary() -> str:
    """Get a summary of everything collected so far.

    Returns:
        Current location, actions remaining, clues seen, and items in inventory.
    """
    return tracker.summary(client.actions_remaining)


# ── Commit decision schema ────────────────────────────────────────────────────

class EvidenceNote(BaseModel):
    clue_id: str
    note: str


class CommitDecision(BaseModel):
    """Your final answer. Returned as structured output and then committed."""

    culprit_id: str
    """EXACT character ID from the briefing cast list (e.g. 'char_silas').
    Do NOT use display names like 'Silas Vane'. Must match char_* format."""

    means_id: str
    """EXACT means ID from the briefing means_options list (e.g. 'means_blade').
    Do NOT use descriptions like 'Fixed-blade knife'. Must match means_* format."""

    evidence_notes: list[EvidenceNote]
    """One note per clue you found — every clue, not just key ones. Each note must
    explain what the clue proves in context (~120 words). Notes on clue IDs you
    never found, or duplicates, are penalised. Non-key found clues are ignored."""

    process_explanation: str
    """Short holistic narrative: who did it, how, and why. NO clue IDs — reference
    evidence by description ('the delivery manifest', 'the monogrammed knife').
    Target ~120 words."""


# ── Agent ─────────────────────────────────────────────────────────────────────

llm = rt.llm.OpenAILLM("gpt-5-nano")

DetectiveAgent = rt.agent_node(
    "Detective Agent",
    tool_nodes=[
        move_to,
        search_location,
        interview_character,
        present_to_character,
        get_investigation_summary,
    ],
    llm=llm,
    system_message=_prompts["detective_agent"]["system_message"],
)

OutputParser = rt.agent_node(
    "Output Parser",
    output_schema=CommitDecision,
    llm=rt.llm.OpenAILLM("gpt-5.2"),
    system_message=_prompts["output_parser"]["system_message"],
)


# ── Flow entry point ──────────────────────────────────────────────────────────
@rt.function_node
async def run_investigation(seed_id: str = "S002") -> dict:
    """Start a session on seed_id, run the agent, and commit the result."""
    session = client.start_session(seed_id)
    tracker.reset(session)
    b = session.briefing

    obs = session.observation
    initial_obs = (
        f"You are in the {obs.name}.\n"
        f"{obs.description}\n"
        f"Characters here: {', '.join(obs.characters_present) or 'nobody'}\n"
        f"Exits: {', '.join(obs.exits)}\n"
        f"Searchable: {obs.searchable}"
    )

    step_lines = "\n".join(
        f"  {s.step_id}: {s.claim}"
        for s in b.timeline_steps
    )

    cast_lines = "\n".join(f"  {m.id}  ({m.name})" for m in b.cast)
    means_lines = "\n".join(f"  {m.id}  ({m.name})" for m in b.means_options)
    location_lines = "\n".join(f"  {loc.id} — {loc.name}" for loc in b.locations)

    prompt = (
        f"═══ CASE BRIEFING ═══\n"
        f"{b.synopsis}\n\n"
        f"Action budget : {session.action_budget} (every action costs 1)\n\n"
        f"══ COMMIT DECISION — USE THESE EXACT IDs ══\n"
        f"culprit_id — pick one:\n"
        f"{cast_lines}\n\n"
        f"means_id — pick one:\n"
        f"{means_lines}\n\n"
        f"Your process_explanation must address each of these claims:\n"
        f"{step_lines}\n\n"
        f"══ LOCATIONS (visit and search all) ══\n"
        f"There are {len(b.locations)} locations. You must visit every one.\n"
        f"Movement is EXIT-BASED — you can only move to rooms listed in your current exits.\n"
        f"To reach a distant room, navigate hop-by-hop; check your exits after each move.\n"
        f"{location_lines}\n\n"
        f"══ CAST (find every character) ══\n"
        f"There are {len(b.cast)} characters. Each is stationary in one room.\n"
        f"You will discover WHERE each character is by visiting rooms and reading\n"
        f"the 'Characters here' line in the move result. Track who you find:\n"
        f"{cast_lines}\n\n"
        f"═══ INITIAL OBSERVATION ═══\n"
        f"{initial_obs}\n\n"
        f"Start your mental map now:\n"
        f"  Current location exits (listed above) — record them before your first move.\n\n"
        f"═══ YOUR TASK ═══\n"
        f"Investigate fully, then commit. Write one evidence note per clue you found (what it\n"
        f"proves in context, ~120 words each), then a short holistic narrative with NO clue IDs\n"
        f"as your process_explanation (~120 words).\n\n"
        f"Begin your investigation now."
    )

    result = await rt.call(DetectiveAgent, prompt)
    result = await rt.call(OutputParser, str(result.text))
    decision: CommitDecision = result.structured

    result = client.commit(
        culprit_id=decision.culprit_id,
        means_id=decision.means_id,
        evidence_notes=[n.model_dump() for n in decision.evidence_notes],
        process_explanation=decision.process_explanation,
    )

    print("\n── Commit result ──────────────────────────────────────")
    if result.score is not None:
        print(f"Score            : {result.score:.4f}")
        print(f"Culprit correct  : {result.culprit_correct}")
        print(f"Means correct    : {result.means_correct}")
        print(f"Notes score      : {result.notes_score:.4f}  "
              f"({result.notes_correct} correct, {result.notes_wrong} wrong, {result.notes_missing} missing)")
        print(f"Explanation score: {result.explanation_score:.4f}")
        print(f"Efficiency bonus : {result.efficiency_score:.4f}")
    else:
        print("Submission accepted. Score revealed at the end of the hackathon.")

    return result.model_dump()


flow = rt.Flow(name="Case Closed Investigation", entry_point=run_investigation)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Case Closed detective agent.")
    parser.add_argument(
        "seed",
        nargs="?",
        default="S001",
        help="Seed to investigate (default: S001).",
    )
    parser.add_argument(
        "--allow-hidden",
        action="store_true",
        help="Allow a hidden H... seed, whose submission is irreversible.",
    )
    args = parser.parse_args()

    if args.seed.upper().startswith("H") and not args.allow_hidden:
        parser.error("Hidden seeds require --allow-hidden")

    result = flow.invoke(args.seed.upper())
