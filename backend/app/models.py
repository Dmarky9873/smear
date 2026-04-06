from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Card(BaseModel):
    id: str
    suit: str | None = None
    rank: str
    is_joker: bool = False


class Player(BaseModel):
    id: int
    name: str
    team_id: int | None = None
    hand: list[Card] = Field(default_factory=list)
    captured_cards: list[Card] = Field(default_factory=list)
    bid: int | None = None
    has_passed: bool = False


class PlayedCard(BaseModel):
    player_id: int
    card: Card


class AuctionHistoryEntry(BaseModel):
    player_id: int
    action: str
    value: int | None = None


class LegalAction(BaseModel):
    type: str
    value: int | None = None
    card_id: str | None = None
    suit: str | None = None


class GameState(BaseModel):
    game_id: str
    phase: str
    players: list[Player]
    current_player_id: int
    dealer_id: int | None = None
    leading_player_id: int | None = None
    winning_bidder_id: int | None = None
    current_bid: int | None = None
    auction_history: list[AuctionHistoryEntry] = Field(default_factory=list)
    trump_suit: str | None = None
    current_trick: list[PlayedCard] = Field(default_factory=list)
    completed_tricks: list[list[PlayedCard]] = Field(default_factory=list)
    scores: dict[str, int] = Field(default_factory=dict)
    round_points: dict[str, int] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)
    legal_actions: list[LegalAction] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class NewGameRequest(BaseModel):
    player_count: int = Field(default=4, ge=2, le=5)
    player_names: list[str] | None = None
    seed: int | None = None
    debug: bool = True


class GameActionRequest(BaseModel):
    type: str
    value: int | None = None
    card_id: str | None = None
    suit: str | None = None


class ResetGameRequest(BaseModel):
    seed: int | None = None


class HealthResponse(BaseModel):
    status: str


class LegalActionsResponse(BaseModel):
    game_id: str
    current_player_id: int
    legal_actions: list[LegalAction] = Field(default_factory=list)


class GameDebugResponse(BaseModel):
    game_id: str
    state: GameState
    metadata: dict[str, Any] = Field(default_factory=dict)
