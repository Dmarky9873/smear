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
    auto_run_bots: bool = True


class CreateLobbyRequest(BaseModel):
    host_name: str
    num_players: int = Field(default=4, ge=3, le=8)
    teams: list[list[int]] | None = None
    host_seat_index: int = Field(default=0, ge=0, le=7)


class JoinLobbyRequest(BaseModel):
    player_name: str
    seat_index: int | None = Field(default=None, ge=0, le=7)


class AddLobbyBotRequest(BaseModel):
    player_token: str
    seat_index: int = Field(ge=0, le=7)
    bot_id: str
    player_name: str | None = None


class RemoveLobbyBotRequest(BaseModel):
    player_token: str
    seat_index: int = Field(ge=0, le=7)


class StartLobbyRequest(BaseModel):
    player_token: str


class LobbyActionRequest(BaseModel):
    player_token: str


class LobbyBidRequest(BaseModel):
    player_token: str
    amount: int = Field(ge=1, le=6)


class LobbyPlayCardRequest(BaseModel):
    player_token: str
    card_code: str


class BidRequest(BaseModel):
    amount: int = Field(ge=1, le=6)
    auto_run_bots: bool = True


class PlayCardRequest(BaseModel):
    card_code: str
    auto_run_bots: bool = True


class DonationCheckoutRequest(BaseModel):
    amount_cents: int = Field(ge=100, le=10000)


class DonationCheckoutResponse(BaseModel):
    url: str


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
    card_count: int
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


class BotProgressResponse(BaseModel):
    active: bool
    player_name: str | None = None
    label: str | None = None
    detail: str | None = None
    completed_units: int | None = None
    total_units: int | None = None
    percent_complete: float | None = None


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


class LearnActionResponse(BaseModel):
    type: Literal["bid", "pass", "play_card"]
    label: str
    amount: int | None = None
    card_code: str | None = None


class LearnChallengeResponse(BaseModel):
    id: str
    phase: Literal["auction", "play"]
    actor_name: str
    prompt: str
    state: GameStateResponse
    options: list[LearnActionResponse]
    best_bot_id: str
    best_bot_label: str
    best_action: LearnActionResponse
    best_action_explanation: str


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
    hidden_cards: list[CardResponse]
    bid_summary: BidSummaryResponse
    awards: dict[str, ScoreAwardResponse]
    results: list[ScoreResultResponse]


class LobbySeatResponse(BaseModel):
    index: int
    player_name: str | None = None
    is_occupied: bool
    is_bot: bool = False
    bot_id: str | None = None
    bot_label: str | None = None
    is_host: bool = False


class LobbyPlayerIdentityResponse(BaseModel):
    player_token: str
    player_name: str
    seat_index: int
    is_host: bool


class LobbyStateResponse(BaseModel):
    code: str
    status: Literal["waiting", "active"]
    num_players: int
    seats: list[LobbySeatResponse]
    teams: list[list[int]] | None = None
    is_full: bool
    you: LobbyPlayerIdentityResponse | None = None
    game_state: GameStateResponse | None = None
    legal_actions: list[PlayCardActionResponse | BidActionResponse | PassActionResponse]
    score: RoundScoreResponse | None = None


class LobbyEventResponse(BaseModel):
    type: Literal["lobby_state"]
    revision: int
    lobby: LobbyStateResponse
