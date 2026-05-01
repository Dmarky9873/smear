from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    ok: bool = True


class NewGameRequest(BaseModel):
    num_players: int = Field(ge=3, le=8)
    player_names: list[str]
    teams: list[list[str]] | None = None


class PlayCardRequest(BaseModel):
    card_code: str


class CardResponse(BaseModel):
    code: str
    rank: str | None
    suit: str | None
    is_joker: bool


class PlayerResponse(BaseModel):
    name: str
    cards: list[CardResponse]
    captured_cards: list[CardResponse]
    captured_count: int


class PlayResponse(BaseModel):
    player_name: str
    card: CardResponse


class TrickStateResponse(BaseModel):
    leader_name: str
    plays: list[PlayResponse]
    trump: str | None
    is_terminal: bool
    winner_name: str | None = None


class TeamResponse(BaseModel):
    constituents: list[str]
    captured_cards: list[CardResponse]
    captured_count: int


class RoundStateResponse(BaseModel):
    players: list[PlayerResponse]
    current_player_name: str
    trump: str | None
    current_trick: TrickStateResponse
    hidden_cards_count: int
    hidden_cards: list[CardResponse]
    trick_history: list[TrickStateResponse]
    teams: list[TeamResponse]
    is_terminal: bool


class GameStateResponse(BaseModel):
    num_players: int
    low: str
    round: RoundStateResponse


class LegalActionResponse(BaseModel):
    type: Literal["play_card"]
    card_code: str


class LegalActionsResponse(BaseModel):
    actions: list[LegalActionResponse]


class ScoreBreakdownResponse(BaseModel):
    high: int
    jack: int
    low: int
    jokers: int
    game: int


class ScoreResultResponse(BaseModel):
    name: str
    member_names: list[str]
    breakdown: ScoreBreakdownResponse
    joker_count: int
    game_total: int
    total_points: int
    captured_cards: list[CardResponse]


class ScoreAwardResponse(BaseModel):
    unit_name: str | None
    player_name: str | None = None
    card: CardResponse | None = None
    game_total: int | None = None
    tied_unit_names: list[str] | None = None
    reason: str | None = None


class RoundScoreResponse(BaseModel):
    trump: str
    high_card: CardResponse
    low_card: CardResponse
    awards: dict[str, ScoreAwardResponse]
    results: list[ScoreResultResponse]
