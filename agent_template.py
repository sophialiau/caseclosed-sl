"""
Case Closed — Railtracks agent template.

Copy this file to your project, set DETECTIVE_TOKEN in .env, and run:

    python agent_template.py

The agent will investigate S001, call game actions as tools, and commit
a structured answer when it's confident.
"""

import yaml
import railtracks as rt
from collections import deque
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


@dataclass(frozen=True)
class EvidenceRecord:
    clue_id: str
    text: str
    source: str
    location_id: str


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
    clues: dict[str, EvidenceRecord] = field(default_factory=dict)
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

    def record_clue(self, clue, source: str) -> None:
        self.clues[clue.clue_id] = EvidenceRecord(
            clue_id=clue.clue_id,
            text=clue.text,
            source=source,
            location_id=self.current_location,
        )

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
            f"  [{record.clue_id}] {record.text} "
            f"(source: {record.source}; location: {record.location_id})"
            for record in sorted(self.clues.values(), key=lambda item: item.clue_id)
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
            f"Presentation refs : use collected clue IDs and inventory item IDs\n"
            f"Known map:\n{chr(10).join(map_lines) or '  none'}\n"
            f"Known characters:\n{chr(10).join(character_lines) or '  none'}\n"
            f"Evidence board:\n{chr(10).join(clue_lines) or '  none'}\n"
            f"Presentations tried:\n{chr(10).join(presentation_lines) or '  none'}"
        )

    def progress_key(self) -> tuple:
        """Compact marker used to detect whether a continuation made progress."""
        return (
            len(self.clues),
            tuple(sorted(self.visited)),
            tuple(sorted(self.searched)),
            tuple(sorted(self.interviewed)),
            tuple(sorted(self.presentations)),
        )

    def location_for_character(self, character_id: str) -> str | None:
        """Return the known location for a character, if discovered."""
        for location_id, characters in self.characters_by_location.items():
            if character_id in characters:
                return location_id
        return None


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
            tracker.record_clue(clue, source=f"search:{evidence.name}")
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
        tracker.record_clue(clue, source=f"interview:{character_id}")
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
        tracker.record_clue(clue, source=f"present:{character_id}<-{ref_id}")
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


def find_path(start: str, goal: str) -> list[str] | None:
    """Find a path through the currently known map."""
    if start == goal:
        return []

    queue = deque([(start, [])])
    seen = {start}
    while queue:
        location, path = queue.popleft()
        for next_location in sorted(tracker.exits_by_location.get(location, set())):
            if next_location in seen:
                continue
            next_path = [*path, next_location]
            if next_location == goal:
                return next_path
            seen.add(next_location)
            queue.append((next_location, next_path))
    return None


def auto_move_to(location_id: str, log: list[str]) -> bool:
    """Move directly to an adjacent location, recording tracker state."""
    if tracker.current_location == location_id:
        return True
    if location_id not in tracker.exits_by_location.get(tracker.current_location, set()):
        return False

    try:
        obs = client.move(location_id)
    except GameError as exc:
        log.append(f"Move blocked to {location_id}: {exc}")
        return False

    tracker.record_location(obs)
    log.append(
        f"Moved to {obs.location_id}; characters={obs.characters_present or 'none'}; "
        f"searchable={obs.searchable}; actions={obs.actions_remaining}"
    )
    return True


def auto_follow_path(path: list[str], log: list[str]) -> bool:
    """Follow a known path, stopping if a move is blocked."""
    for location_id in path:
        if not auto_move_to(location_id, log):
            return False
    return True


def auto_search_current(log: list[str]) -> bool:
    """Search current location when useful."""
    if tracker.current_location not in tracker.searchable_locations:
        return False
    if tracker.current_location in tracker.searched:
        return False

    try:
        obs = client.search()
    except GameError as exc:
        log.append(f"Search blocked at {tracker.current_location}: {exc}")
        return False

    tracker.searched.add(tracker.current_location)
    found_bits = []
    for evidence in obs.found:
        if evidence.item_id:
            tracker.inventory.add(evidence.item_id)
        clue_ids = []
        for clue in evidence.yields:
            tracker.record_clue(clue, source=f"search:{evidence.name}")
            clue_ids.append(clue.clue_id)
        found_bits.append(f"{evidence.name}({', '.join(clue_ids) or 'no clues'})")
    log.append(
        f"Searched {tracker.current_location}: {found_bits or 'nothing'}; "
        f"actions={obs.actions_remaining}"
    )
    return True


def auto_interview_here(log: list[str]) -> bool:
    """Interview every not-yet-interviewed character at the current location."""
    progressed = False
    for character_id in sorted(tracker.characters_by_location.get(tracker.current_location, set())):
        if character_id in tracker.interviewed:
            continue
        try:
            obs = client.interview(character_id)
        except GameError as exc:
            log.append(f"Interview blocked for {character_id}: {exc}")
            continue

        tracker.interviewed.add(character_id)
        for clue in obs.said:
            tracker.record_clue(clue, source=f"interview:{character_id}")
        log.append(
            f"Interviewed {character_id}: {[clue.clue_id for clue in obs.said] or 'nothing'}; "
            f"actions={obs.actions_remaining}"
        )
        progressed = True
    return progressed


def auto_primary_sweep(reserve_actions: int = 18, max_steps: int = 80) -> str:
    """Deterministically collect cheap search/interview evidence before LLM reasoning."""
    log = ["AUTO PRIMARY SWEEP START"]
    blocked: dict[str, tuple[str, ...]] = {}

    for _ in range(max_steps):
        if client.actions_remaining <= reserve_actions:
            break

        progressed = False
        progressed = auto_search_current(log) or progressed
        progressed = auto_interview_here(log) or progressed
        if progressed:
            continue

        inventory_key = tuple(sorted(tracker.inventory))
        reachable_unvisited = []
        for location in sorted(tracker.all_locations - tracker.visited):
            if blocked.get(location) == inventory_key:
                continue
            path = find_path(tracker.current_location, location)
            if path is not None:
                reachable_unvisited.append((len(path), location, path))

        if reachable_unvisited:
            _, location, path = min(reachable_unvisited)
            if not auto_follow_path(path, log):
                blocked[location] = inventory_key
            continue

        reachable_unsearched = []
        for location in sorted(tracker.searchable_locations - tracker.searched):
            path = find_path(tracker.current_location, location)
            if path is not None:
                reachable_unsearched.append((len(path), location, path))

        if reachable_unsearched:
            _, _, path = min(reachable_unsearched)
            auto_follow_path(path, log)
            continue

        reachable_missing_interviews = []
        for location, characters in tracker.characters_by_location.items():
            if not (characters - tracker.interviewed):
                continue
            path = find_path(tracker.current_location, location)
            if path is not None:
                reachable_missing_interviews.append((len(path), location, path))

        if reachable_missing_interviews:
            _, _, path = min(reachable_missing_interviews)
            auto_follow_path(path, log)
            continue

        break

    log.append("AUTO PRIMARY SWEEP END")
    log.append(tracker.summary(client.actions_remaining))
    return "\n".join(log)


def character_terms(character) -> set[str]:
    """Names/initials used to connect clues to a character."""
    name = character.name.lower()
    parts = [part.strip(".,'\"()[]") for part in name.split() if part.strip(".,'\"()[]")]
    terms = set(parts)
    if parts:
        terms.add(name)
        terms.add(".".join(part[0] for part in parts if part) + ".")
        terms.add("".join(part[0] for part in parts if part))
    return {term for term in terms if len(term) > 1}


def clue_mentions_character(record: EvidenceRecord, terms: set[str]) -> bool:
    """Return whether a clue text appears connected to a character."""
    text = record.text.lower()
    compact = text.replace(" ", "")
    return any(term in text or term.replace(".", "") in compact for term in terms)


def auto_present(character_id: str, ref_id: str, log: list[str]) -> bool:
    """Present a collected ref to a character if possible."""
    if ref_id not in tracker.clues and ref_id not in tracker.inventory:
        return False
    attempt = (character_id, ref_id)
    if attempt in tracker.presentations:
        return False

    location_id = tracker.location_for_character(character_id)
    if not location_id:
        return False
    path = find_path(tracker.current_location, location_id)
    if path is None:
        return False
    if not auto_follow_path(path, log):
        return False

    try:
        obs = client.present(character_id, ref_id)
    except GameError as exc:
        log.append(f"Present blocked {character_id} <- {ref_id}: {exc}")
        return False

    tracker.presentations.add(attempt)
    for clue in obs.revealed:
        tracker.record_clue(clue, source=f"present:{character_id}<-{ref_id}")
    log.append(
        f"Presented {character_id} <- {ref_id}: "
        f"{[clue.clue_id for clue in obs.revealed] or 'no unlock'}; "
        f"actions={obs.actions_remaining}"
    )
    return bool(obs.revealed)


def auto_targeted_presentations(briefing, max_attempts: int = 12, reserve_actions: int = 8) -> str:
    """Try high-signal guarded-testimony unlocks without brute-forcing every pair."""
    log = ["AUTO TARGETED PRESENTATIONS START"]
    attempts = 0

    for character in briefing.cast:
        if attempts >= max_attempts or client.actions_remaining <= reserve_actions:
            break
        character_id = character.id
        if not tracker.location_for_character(character_id):
            continue

        refs: list[str] = []
        refs.extend(sorted(tracker.inventory))
        terms = character_terms(character)
        refs.extend(
            record.clue_id
            for record in sorted(tracker.clues.values(), key=lambda item: item.clue_id)
            if clue_mentions_character(record, terms)
        )

        seen_refs = set()
        for ref_id in refs:
            if attempts >= max_attempts or client.actions_remaining <= reserve_actions:
                break
            if ref_id in seen_refs:
                continue
            seen_refs.add(ref_id)
            if (character_id, ref_id) in tracker.presentations:
                continue
            auto_present(character_id, ref_id, log)
            attempts += 1

    log.append("AUTO TARGETED PRESENTATIONS END")
    log.append(tracker.summary(client.actions_remaining))
    return "\n".join(log)


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


def word_count(text: str) -> int:
    """Return a simple whitespace word count."""
    return len(text.split())


def limit_words(text: str, max_words: int = 140) -> str:
    """Trim text to max_words while keeping a readable ending."""
    words = text.split()
    if len(words) <= max_words:
        return text

    clipped = " ".join(words[:max_words]).rstrip(" ,;:-")
    if clipped and clipped[-1] not in ".!?":
        clipped += "."
    return clipped


def enforce_word_limits(decision: CommitDecision, max_words: int = 140) -> None:
    """Apply hard word-count guardrails before local validation and commit."""
    for note in decision.evidence_notes:
        note.note = limit_words(note.note, max_words)
    decision.process_explanation = limit_words(decision.process_explanation, max_words)


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


def validate_decision(decision: CommitDecision, briefing: object) -> list[str]:
    """Return commit-blocking errors for IDs the server would score as invalid."""
    errors = []
    cast_ids = {character.id for character in briefing.cast}
    means_ids = {means.id for means in briefing.means_options}
    note_ids = [note.clue_id for note in decision.evidence_notes]

    if decision.culprit_id not in cast_ids:
        errors.append(f"Unknown culprit_id: {decision.culprit_id}")
    if decision.means_id not in means_ids:
        errors.append(f"Unknown means_id: {decision.means_id}")
    if len(note_ids) != len(set(note_ids)):
        errors.append("Duplicate clue IDs in evidence_notes")

    unknown_clues = sorted(set(note_ids) - set(tracker.clues))
    if unknown_clues:
        errors.append(f"Evidence notes contain uncollected clues: {unknown_clues}")

    long_notes = [
        f"{note.clue_id} ({word_count(note.note)} words)"
        for note in decision.evidence_notes
        if word_count(note.note) > 140
    ]
    if long_notes:
        errors.append(f"Evidence notes exceed 140 words: {long_notes}")

    explanation_words = word_count(decision.process_explanation)
    if explanation_words > 140:
        errors.append(f"Process explanation is {explanation_words} words; maximum is 140")
    return errors


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

    auto_sweep_log = auto_primary_sweep(reserve_actions=16)
    auto_presentation_log = auto_targeted_presentations(b)

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
        f"══ LOCATIONS (known map targets) ══\n"
        f"There are {len(b.locations)} locations. Complete a cheap primary sweep before committing.\n"
        f"Movement is EXIT-BASED — you can only move to rooms listed in your current exits.\n"
        f"To reach a distant room, navigate hop-by-hop; check your exits after each move.\n"
        f"{location_lines}\n\n"
        f"══ CAST (possible suspects and witnesses) ══\n"
        f"There are {len(b.cast)} characters. Each is stationary in one room.\n"
        f"You will discover WHERE each character is by visiting rooms and reading\n"
        f"the 'Characters here' line in the move result. Track who you find:\n"
        f"{cast_lines}\n\n"
        f"═══ INITIAL OBSERVATION ═══\n"
        f"{initial_obs}\n\n"
        f"Start your mental map now:\n"
        f"  Current location exits (listed above) — record them before your first move.\n\n"
        f"═══ AUTO PRIMARY SWEEP ALREADY COMPLETED ═══\n"
        f"{auto_sweep_log}\n\n"
        f"═══ YOUR TASK ═══\n"
        f"The cheap search/interview sweep has already been done for you. Build the cheapest\n"
        f"strong case from the current evidence board. Use tools only for missing cheap work\n"
        f"or for a small budget of 3-6 targeted presentations if actions allow: present\n"
        f"inventory item IDs first to the character named by, owning, signing, accessing, or\n"
        f"contradicted by that item. Do not revisit searched rooms, re-interview characters,\n"
        f"or exhaust speculative presentations once culprit, means, motive, opportunity, and\n"
        f"timeline are supported.\n"
        f"Write one evidence note per clue you found (what it proves in context, ~120 words\n"
        f"each), then a short holistic narrative with NO clue IDs as your process_explanation\n"
        f"(~120 words).\n\n"
        f"Begin your investigation now."
    )

    handoffs: list[str] = [
        "AUTOMATED INVESTIGATION HANDOFF:\n"
        "The primary sweep and targeted presentation pass are complete. Use the "
        "authoritative evidence board and timeline claims to choose the best culprit_id, "
        "means_id, evidence notes, and process explanation.\n\n"
        f"{auto_sweep_log}\n\n{auto_presentation_log}"
    ]

    evidence_lines = "\n".join(
        f"[{record.clue_id}] {record.text} "
        f"(source: {record.source}; location: {record.location_id})"
        for record in sorted(tracker.clues.values(), key=lambda item: item.clue_id)
    )
    parser_prompt = (
        f"{chr(10).join(handoffs)}\n\n"
        f"AUTHORITATIVE EVIDENCE BOARD:\n{evidence_lines or 'none'}\n\n"
        f"TIMELINE CLAIMS:\n{step_lines}\n\n"
        f"AUTHORITATIVE COLLECTED CLUE IDS: {sorted(tracker.clues)}\n"
        f"VALID CULPRIT IDS: {sorted(character.id for character in b.cast)}\n"
        f"VALID MEANS IDS: {sorted(means.id for means in b.means_options)}"
    )
    result = await rt.call(OutputParser, parser_prompt)
    decision: CommitDecision = result.structured
    enforce_word_limits(decision)

    validation_errors = validate_decision(decision, b)
    if validation_errors:
        formatted = "\n".join(f"- {error}" for error in validation_errors)
        raise RuntimeError(f"Commit blocked by local validation:\n{formatted}")

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
