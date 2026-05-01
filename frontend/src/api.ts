import type {
  GameState,
  LegalActionsResponse,
  ReadyBotListResponse,
  Score,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? String(payload.detail)
        : `${response.status} ${response.statusText}`;
    throw new Error(detail);
  }

  return payload as T;
}

export type NewGamePayload = {
  num_players: number;
  player_names: string[];
  teams: string[][] | null;
  player_bots: (string | null)[];
};

export function fetchBots(): Promise<ReadyBotListResponse> {
  return request<ReadyBotListResponse>("/bots");
}

export function createGame(payload: NewGamePayload): Promise<GameState> {
  return request<GameState>("/game/new", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function resetRound(): Promise<GameState> {
  return request<GameState>("/game/reset", {
    method: "POST",
  });
}

export function nextRound(): Promise<GameState> {
  return request<GameState>("/game/next-round", {
    method: "POST",
  });
}

export function placeBid(amount: number): Promise<GameState> {
  return request<GameState>("/game/auction/bid", {
    method: "POST",
    body: JSON.stringify({ amount }),
  });
}

export function passAuction(): Promise<GameState> {
  return request<GameState>("/game/auction/pass", {
    method: "POST",
  });
}

export function fetchGameState(): Promise<GameState> {
  return request<GameState>("/game/state");
}

export function fetchLegalActions(): Promise<LegalActionsResponse> {
  return request<LegalActionsResponse>("/game/legal-actions");
}

export function playCard(cardCode: string): Promise<GameState> {
  return request<GameState>("/game/play", {
    method: "POST",
    body: JSON.stringify({ card_code: cardCode }),
  });
}

export function fetchScore(): Promise<Score> {
  return request<Score>("/game/score");
}
