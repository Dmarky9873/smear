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
  round: RoundState;
};

export type LegalAction = {
  type: "play_card";
  card_code: string;
};

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
