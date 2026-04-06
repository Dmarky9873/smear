export type Card = {
  id: string;
  suit: string | null;
  rank: string;
  is_joker: boolean;
};

export type Player = {
  id: number;
  name: string;
  team_id: number | null;
  hand: Card[];
  captured_cards: Card[];
  bid: number | null;
  has_passed: boolean;
};

export type PlayedCard = {
  player_id: number;
  card: Card;
};

export type AuctionHistoryEntry = {
  player_id: number;
  action: string;
  value: number | null;
};

export type LegalAction = {
  type: string;
  value: number | null;
  card_id: string | null;
  suit: string | null;
};

export type GameState = {
  game_id: string;
  phase: string;
  players: Player[];
  current_player_id: number;
  dealer_id: number | null;
  leading_player_id: number | null;
  winning_bidder_id: number | null;
  current_bid: number | null;
  auction_history: AuctionHistoryEntry[];
  trump_suit: string | null;
  current_trick: PlayedCard[];
  completed_tricks: PlayedCard[][];
  scores: Record<string, number>;
  round_points: Record<string, number>;
  logs: string[];
  legal_actions: LegalAction[];
  debug: Record<string, unknown>;
};

export type NewGameRequest = {
  player_count?: number;
  player_names?: string[];
  seed?: number;
  debug?: boolean;
};

export type ResetGameRequest = {
  seed?: number;
};

export type LegalActionsResponse = {
  game_id: string;
  current_player_id: number;
  legal_actions: LegalAction[];
};

export type GameDebugResponse = {
  game_id: string;
  state: GameState;
  metadata: Record<string, unknown>;
};
