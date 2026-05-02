import type {
  BotProgress,
  GameState,
  LegalActionsResponse,
  ReadyBotListResponse,
  Score,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type MutatingRequestOptions = {
  auto_run_bots?: boolean;
};

function withAutoRunBots(
  path: string,
  options?: MutatingRequestOptions,
): string {
  if (options?.auto_run_bots === undefined) {
    return path;
  }

  const searchParams = new URLSearchParams({
    auto_run_bots: String(options.auto_run_bots),
  });
  return `${path}?${searchParams.toString()}`;
}

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
  auto_run_bots?: boolean;
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

export function resetRound(options?: MutatingRequestOptions): Promise<GameState> {
  return request<GameState>(withAutoRunBots("/game/reset", options), {
    method: "POST",
  });
}

export function nextRound(options?: MutatingRequestOptions): Promise<GameState> {
  return request<GameState>(withAutoRunBots("/game/next-round", options), {
    method: "POST",
  });
}

export function placeBid(
  amount: number,
  options?: MutatingRequestOptions,
): Promise<GameState> {
  return request<GameState>("/game/auction/bid", {
    method: "POST",
    body: JSON.stringify({
      amount,
      auto_run_bots: options?.auto_run_bots,
    }),
  });
}

export function passAuction(
  options?: MutatingRequestOptions,
): Promise<GameState> {
  return request<GameState>(withAutoRunBots("/game/auction/pass", options), {
    method: "POST",
  });
}

export function fetchGameState(): Promise<GameState> {
  return request<GameState>("/game/state");
}

export function fetchLegalActions(): Promise<LegalActionsResponse> {
  return request<LegalActionsResponse>("/game/legal-actions");
}

export function playCard(
  cardCode: string,
  options?: MutatingRequestOptions,
): Promise<GameState> {
  return request<GameState>("/game/play", {
    method: "POST",
    body: JSON.stringify({
      card_code: cardCode,
      auto_run_bots: options?.auto_run_bots,
    }),
  });
}

export function fetchScore(): Promise<Score> {
  return request<Score>("/game/score");
}

export function stepBotTurn(): Promise<GameState> {
  return request<GameState>("/game/bots/step", {
    method: "POST",
  });
}

export function fetchBotProgress(): Promise<BotProgress> {
  return request<BotProgress>("/game/bots/progress");
}
