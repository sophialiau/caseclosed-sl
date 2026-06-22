"""
Case Closed — Python client library.

Load your team token from .env and go:

    from detective_client import DetectiveClient

    client = DetectiveClient()          # reads DETECTIVE_TOKEN from .env
    session = client.start_session("S001")
    obs = client.move("loc_office")
    obs = client.search()
    obs = client.interview("char_finn")
    obs = client.present("char_mara", ref_id="item_ledger")
    result = client.commit(
        culprit_id="char_silas",
        means_id="means_blade",
        evidence_notes=[
            {"clue_id": "clue_04", "note": "Silas signed the midnight delivery himself, placing him at the warehouse."},
            {"clue_id": "clue_S2", "note": "He confessed when confronted with the ledger and the knife."},
        ],
        process_explanation="Silas arranged the cargo run to lure Marsh, then stabbed him when Marsh threatened to expose the operation.",
    )
"""

from __future__ import annotations

import os
import requests
from pydantic import BaseModel, Field


class GameError(Exception):
    """Raised when a game action is rejected (wrong location, bad ref_id, etc.)."""

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set env vars manually if not installed


# ── Shared leaf models ────────────────────────────────────────────────────────

class ClueText(BaseModel):
    """A single clue ID paired with its text."""
    clue_id: str
    text: str


# ── Session start ─────────────────────────────────────────────────────────────

class LocationInfo(BaseModel):
    id: str
    name: str


class CastMember(BaseModel):
    id: str
    name: str


class MeansOption(BaseModel):
    id: str
    name: str


class TimelineStepBrief(BaseModel):
    step_id: str
    claim: str


class Briefing(BaseModel):
    synopsis: str
    locations: list[LocationInfo]
    cast: list[CastMember]
    means_options: list[MeansOption]
    timeline_steps: list[TimelineStepBrief] = []


class LocationObservation(BaseModel):
    location_id: str
    name: str
    description: str
    characters_present: list[str]
    exits: list[str]
    searchable: bool


class SessionInfo(BaseModel):
    seed_id: str
    briefing: Briefing
    start_location: str
    action_budget: int
    actions_remaining: int
    session_id: str
    observation: LocationObservation


# ── State ──────────────────────────────────────────────────────────────────────

class GameState(BaseModel):
    current_location: str
    actions_remaining: int


# ── Action observations ────────────────────────────────────────────────────────

class MoveObservation(BaseModel):
    ok: bool = True
    location_id: str
    name: str
    description: str
    characters_present: list[str]
    exits: list[str]
    searchable: bool
    actions_remaining: int


class EvidenceFound(BaseModel):
    evidence_id: str
    name: str
    yields: list[ClueText]
    is_item: bool
    item_id: str | None = None


class SearchObservation(BaseModel):
    ok: bool = True
    found: list[EvidenceFound]
    actions_remaining: int


class InterviewObservation(BaseModel):
    ok: bool = True
    character_id: str
    name: str
    said: list[ClueText]
    actions_remaining: int


class PresentObservation(BaseModel):
    ok: bool = True
    accepted: bool
    revealed: list[ClueText]
    actions_remaining: int


# ── Commit ─────────────────────────────────────────────────────────────────────

class CommitResult(BaseModel):
    ok: bool = True
    submission_id: str
    actions_remaining: int
    process_explanation: str
    accepted: bool = False
    # Practice seeds return all fields; scored seeds return only accepted=True until reveal:
    score: float | None = None
    culprit_correct: bool | None = None
    means_correct: bool | None = None
    notes_score: float | None = None
    explanation_score: float | None = None
    notes_correct: int | None = None
    notes_wrong: int | None = None
    notes_missing: int | None = None
    efficiency_score: float | None = None
    actions_used: int | None = None


# ── Seeds ──────────────────────────────────────────────────────────────────────

class SeedInfo(BaseModel):
    seed_id: str
    visibility: str
    action_budget: int


class SeedsResponse(BaseModel):
    seeds: list[SeedInfo]


# ── Client ─────────────────────────────────────────────────────────────────────

class DetectiveClient:
    """Typed HTTP client for the Case Closed detective API.

    Args:
        token: Bearer token for your team. If omitted, reads DETECTIVE_TOKEN from env.
        base_url: API base URL. If omitted, reads DETECTIVE_BASE_URL from env or
                  defaults to http://localhost:8000.
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.token = token or os.environ["DETECTIVE_TOKEN"]
        self.base_url = (base_url or os.environ.get("DETECTIVE_BASE_URL", "https://railtracks-case-closed.azurewebsites.net/")).rstrip("/")
        self.session_id: str | None = None
        # Client-side knowledge tracking — updated from each action response
        self.clues_seen: set[str] = set()
        self.inventory: set[str] = set()
        self.actions_remaining: int = 0
        self.current_location: str = ""

    # ── Internal HTTP ──────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _get(self, path: str) -> dict:
        r = requests.get(f"{self.base_url}{path}", headers=self._headers())
        self._raise(r)
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base_url}{path}", json=body, headers=self._headers())
        self._raise(r)
        return r.json()

    @staticmethod
    def _raise(r: requests.Response) -> None:
        if r.ok:
            return
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise requests.exceptions.HTTPError(
            f"HTTP {r.status_code}: {detail}", response=r
        )

    def _act(self, verb: str, args: dict) -> dict:
        if not self.session_id:
            raise RuntimeError("No active session — call start_session() first")
        r = requests.post(
            f"{self.base_url}/sessions/{self.session_id}/act",
            json={"verb": verb, "args": args},
            headers=self._headers(),
        )
        if r.status_code == 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise GameError(detail)
        self._raise(r)
        return r.json()

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def start_session(self, seed_id: str) -> SessionInfo:
        """Start a new investigation session. Resets all client-side tracking."""
        data = self._post("/sessions", {"seed_id": seed_id})
        result = SessionInfo.model_validate(data)
        self.session_id = result.session_id
        self.actions_remaining = result.actions_remaining
        self.current_location = result.start_location
        self.clues_seen = set()
        self.inventory = set()
        return result

    def get_state(self) -> GameState:
        """Fetch authoritative state from the server (location + actions remaining)."""
        if not self.session_id:
            raise RuntimeError("No active session")
        data = self._get(f"/sessions/{self.session_id}/state")
        gs = GameState.model_validate(data)
        self.actions_remaining = gs.actions_remaining
        self.current_location = gs.current_location
        return gs

    # ── Actions ────────────────────────────────────────────────────────────────

    def move(self, location_id: str) -> MoveObservation:
        """Move to an adjacent location. Costs 1 action."""
        data = self._act("move", {"location_id": location_id})
        result = MoveObservation.model_validate(data)
        self.actions_remaining = result.actions_remaining
        self.current_location = result.location_id
        return result

    def search(self) -> SearchObservation:
        """Search the current location for evidence and items. Costs 1 action."""
        data = self._act("search", {})
        result = SearchObservation.model_validate(data)
        self.actions_remaining = result.actions_remaining
        for ev in result.found:
            for clue in ev.yields:
                self.clues_seen.add(clue.clue_id)
            if ev.item_id:
                self.inventory.add(ev.item_id)
        return result

    def interview(self, character_id: str) -> InterviewObservation:
        """Talk to a character at the current location. Returns their freely offered clues.
        Repeating costs an action but gives the same response. Costs 1 action."""
        data = self._act("interview", {"character_id": character_id})
        result = InterviewObservation.model_validate(data)
        self.actions_remaining = result.actions_remaining
        for clue in result.said:
            self.clues_seen.add(clue.clue_id)
        return result

    def present(self, character_id: str, ref_id: str) -> PresentObservation:
        """Present an item or clue to a character to unlock their guarded testimony.

        ref_id is either an item_id (from inventory) or a clue_id (from clues_seen).
        The server determines which type based on the character's unlock conditions.
        Costs 1 action.
        """
        data = self._act("present", {"character_id": character_id, "ref_id": ref_id})
        result = PresentObservation.model_validate(data)
        self.actions_remaining = result.actions_remaining
        for clue in result.revealed:
            self.clues_seen.add(clue.clue_id)
        return result

    def commit(
        self,
        culprit_id: str,
        means_id: str,
        evidence_notes: list[dict],
        process_explanation: str,
    ) -> CommitResult:
        """Submit your final answer. Irreversible on hidden seeds.

        Args:
            culprit_id: Exact character ID of the killer (e.g. 'char_silas').
            means_id: Exact means ID (e.g. 'means_blade').
            evidence_notes: One note per clue you found. Each element must be a dict
                with exactly two keys: 'clue_id' and 'note'. The note should explain
                what the clue proves in context (~120 words). Submit notes for every
                clue collected — non-key clues are silently ignored; notes on clue IDs
                you never found, or duplicates, count as wrong and are penalised.
            process_explanation: Short holistic narrative — who did it, how, and why.
                No clue IDs. Target ~120 words.
        """
        args = {
            "culprit_id": culprit_id,
            "means_id": means_id,
            "evidence_notes": evidence_notes,
            "process_explanation": process_explanation,
        }
        data = self._act("commit", args)
        return CommitResult.model_validate(data)

    # ── Meta ───────────────────────────────────────────────────────────────────

    def list_seeds(self) -> SeedsResponse:
        """List all available seeds and their visibility."""
        return SeedsResponse.model_validate(self._get("/seeds"))

    # ── Submission helpers ─────────────────────────────────────────────────────

    def knowledge_summary(self) -> str:
        """Return a human-readable summary of current investigation state."""
        return (
            f"Location          : {self.current_location}\n"
            f"Actions remaining : {self.actions_remaining}\n"
            f"Clues collected   : {sorted(self.clues_seen) or 'none'}\n"
            f"Items in inventory: {sorted(self.inventory) or 'none'}"
        )
