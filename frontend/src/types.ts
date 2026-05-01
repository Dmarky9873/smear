export type Card = {
  code: string;
  rank: string | null;
  suit: string | null;
  is_joker: boolean;
};

export type Player = {
  name: string;
  cards: Card[];
  captured_cards: Card[];
  captured_count: number;
};

export type Play = {
  player_name: string;
  card: Card;
};

export type TrickState = {
  leader_name: string;
  plays: Play[];
  trump: string | null;
  is_terminal: boolean;
  winner_name: string | null;
};

export type Team = {
  constituents: string[];
  captured_cards: Card[];
  captured_count: number;
};

export type RoundState = {
  players: Player[];
  current_player_name: string;
  trump: string | null;
  current_trick: TrickState;
  hidden_cards_count: number;
  hidden_cards: Card[];
  trick_history: TrickState[];
  teams: Team[];
  is_terminal: boolean;
};

export type GameState = {
  num_players: number;
  low: string;
  phase: GamePhase;
  auction: AuctionState;
  round: RoundState;
};

export type GamePhase = "auction" | "play" | "round_complete";

export type AuctionEvent = {
  bidder_name: string;
  action: "bid" | "pass";
  amount: number | null;
};

export type AuctionState = {
  dealer_name: string;
  current_bidder_name: string;
  current_high_bid: number | null;
  highest_bidder_name: string | null;
  passed_player_names: string[];
  active_player_names: string[];
  bid_history: AuctionEvent[];
  is_complete: boolean;
};

export type PlayCardAction = {
  type: "play_card";
  card_code: string;
};

export type BidAction = {
  type: "bid";
  amount: number;
};

export type PassAction = {
  type: "pass";
};

export type LegalAction = PlayCardAction | BidAction | PassAction;

export type LegalActionsResponse = {
  actions: LegalAction[];
};

export type ScoreBreakdown = {
  high: number;
  jack: number;
  low: number;
  jokers: number;
  game: number;
};

export type ScoreAward = {
  unit_name: string | null;
  player_name: string | null;
  card: Card | null;
  game_total: number | null;
  tied_unit_names: string[] | null;
  reason: string | null;
};

export type ScoreResult = {
  name: string;
  member_names: string[];
  breakdown: ScoreBreakdown;
  joker_count: number;
  game_total: number;
  total_points: number;
  captured_cards: Card[];
};

export type Score = {
  trump: string;
  high_card: Card;
  low_card: Card;
  awards: {
    high: ScoreAward;
    jack: ScoreAward;
    low: ScoreAward;
    game: ScoreAward;
  };
  results: ScoreResult[];
};
