from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    ok: bool = True


class NewGameRequest(BaseModel):
    num_players: int = Field(ge=3, le=8)
    player_names: list[str]
    teams: list[list[str]] | None = None
    player_bots: list[str | None] | None = None


class BidRequest(BaseModel):
    amount: int = Field(ge=1, le=6)


class PlayCardRequest(BaseModel):
    card_code: str


class CardResponse(BaseModel):
    code: str
    rank: str | None
    suit: str | None
    is_joker: bool


class PlayerResponse(BaseModel):
    name: str
    bot_id: str | None = None
    bot_label: str | None = None
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


class AuctionEventResponse(BaseModel):
    bidder_name: str
    action: Literal["bid", "pass"]
    amount: int | None = None


class AuctionStateResponse(BaseModel):
    dealer_name: str
    current_bidder_name: str
    current_high_bid: int | None
    highest_bidder_name: str | None
    passed_player_names: list[str]
    active_player_names: list[str]
    bid_history: list[AuctionEventResponse]
    is_complete: bool


class MatchScoreEntryResponse(BaseModel):
    name: str
    points: int


class MatchStateResponse(BaseModel):
    round_number: int
    target_score: int
    scores: list[MatchScoreEntryResponse]
    is_complete: bool
    winner_names: list[str]


class GameStateResponse(BaseModel):
    num_players: int
    low: str
    phase: Literal["auction", "play", "round_complete", "match_complete"]
    auction: AuctionStateResponse
    match: MatchStateResponse
    round: RoundStateResponse


class ReadyBotResponse(BaseModel):
    id: str
    label: str
    description: str


class ReadyBotListResponse(BaseModel):
    bots: list[ReadyBotResponse]


class PlayCardActionResponse(BaseModel):
    type: Literal["play_card"]
    card_code: str


class BidActionResponse(BaseModel):
    type: Literal["bid"]
    amount: int


class PassActionResponse(BaseModel):
    type: Literal["pass"]


class LegalActionsResponse(BaseModel):
    actions: list[PlayCardActionResponse | BidActionResponse | PassActionResponse]


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
    match_delta: int
    bid_amount: int | None = None
    made_bid: bool | None = None
    captured_cards: list[CardResponse]


class ScoreAwardResponse(BaseModel):
    unit_name: str | None
    player_name: str | None = None
    card: CardResponse | None = None
    game_total: int | None = None
    tied_unit_names: list[str] | None = None
    reason: str | None = None


class BidSummaryResponse(BaseModel):
    bidder_name: str | None
    unit_name: str | None
    amount: int | None
    points_won: int | None
    made_bid: bool | None
    match_delta: int | None


class RoundScoreResponse(BaseModel):
    trump: str
    high_card: CardResponse
    low_card: CardResponse
    bid_summary: BidSummaryResponse
    awards: dict[str, ScoreAwardResponse]
    results: list[ScoreResultResponse]
