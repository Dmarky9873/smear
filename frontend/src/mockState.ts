import type { GameState, NewGameRequest } from "./types";

export const defaultNewGameRequest: NewGameRequest = {
  player_count: 4,
  player_names: ["North", "East", "South", "West"],
  debug: true,
};

export const emptyGameState: GameState = {
  game_id: "",
  phase: "not_started",
  players: [],
  current_player_id: 0,
  dealer_id: null,
  leading_player_id: null,
  winning_bidder_id: null,
  current_bid: null,
  auction_history: [],
  trump_suit: null,
  current_trick: [],
  completed_tricks: [],
  scores: {},
  round_points: {},
  logs: [],
  legal_actions: [],
  debug: {},
};
