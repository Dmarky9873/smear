import type {
  GameDebugResponse,
  GameState,
  LegalAction,
  LegalActionsResponse,
  NewGameRequest,
  ResetGameRequest,
} from "./types";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as
      | { detail?: string }
      | null;
    throw new Error(payload?.detail ?? `Request failed with ${response.status}`);
  }

  return (await response.json()) as T;
}

export function createNewGame(payload: NewGameRequest): Promise<GameState> {
  return request<GameState>("/game/new", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getGameState(gameId: string): Promise<GameState> {
  return request<GameState>(`/game/${gameId}`);
}

export function getLegalActions(gameId: string): Promise<LegalActionsResponse> {
  return request<LegalActionsResponse>(`/game/${gameId}/legal-actions`);
}

export function applyGameAction(gameId: string, action: LegalAction): Promise<GameState> {
  return request<GameState>(`/game/${gameId}/action`, {
    method: "POST",
    body: JSON.stringify(action),
  });
}

export function resetGame(gameId: string, payload?: ResetGameRequest): Promise<GameState> {
  return request<GameState>(`/game/${gameId}/reset`, {
    method: "POST",
    body: JSON.stringify(payload ?? {}),
  });
}

export function runAiMove(gameId: string): Promise<GameState> {
  return request<GameState>(`/game/${gameId}/ai-move`, {
    method: "POST",
  });
}

export function getGameDebug(gameId: string): Promise<GameDebugResponse> {
  return request<GameDebugResponse>(`/game/${gameId}/debug`);
}
