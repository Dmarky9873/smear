import type {
  BotProgress,
  GameState,
  LegalActionsResponse,
  LearnChallenge,
  LobbyState,
  ReadyBotListResponse,
  Score,
} from "./types";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_SESSION_STORAGE_KEY = "smear-session-id";

export type MutatingRequestOptions = {
  auto_run_bots?: boolean;
};

export type NewGamePayload = {
  num_players: number;
  player_names: string[];
  teams: string[][] | null;
  player_bots: (string | null)[];
  auto_run_bots?: boolean;
};

export type CreateLobbyPayload = {
  host_name: string;
  num_players: number;
  teams: number[][] | null;
  host_seat_index?: number;
};

export type JoinLobbyPayload = {
  player_name: string;
  seat_index?: number | null;
};

export type DonationCheckoutPayload = {
  amount_cents: number;
};

export type DonationCheckoutResponse = {
  url: string;
};

export type ApiClientOptions = {
  apiBaseUrl?: string;
  sessionId?: string | null;
};

export type GameStateEvent = {
  type: "game_state";
  revision: number;
  state: GameState | null;
};

export type LobbyStateEvent = {
  type: "lobby_state";
  revision: number;
  lobby: LobbyState;
};

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

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

function buildGameEventsUrl(
  apiBaseUrl: string,
  sessionId: string | null | undefined,
): string {
  const url = new URL(apiBaseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `${url.pathname.replace(/\/$/, "")}/game/ws`;
  url.search = "";
  if (sessionId) {
    url.searchParams.set("session_id", sessionId);
  }
  return url.toString();
}

function buildLobbyEventsUrl(
  apiBaseUrl: string,
  lobbyCode: string,
  playerToken: string,
): string {
  const url = new URL(apiBaseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `${url.pathname.replace(/\/$/, "")}/lobbies/${encodeURIComponent(
    lobbyCode,
  )}/ws`;
  url.search = "";
  url.searchParams.set("player_token", playerToken);
  return url.toString();
}

function buildRequestHeaders(
  sessionId: string | null | undefined,
  headers?: HeadersInit,
): HeadersInit {
  return {
    "Content-Type": "application/json",
    ...(sessionId ? { "X-Smear-Session-Id": sessionId } : {}),
    ...(headers ?? {}),
  };
}

async function request<T>(
  apiBaseUrl: string,
  sessionId: string | null | undefined,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: buildRequestHeaders(sessionId, init?.headers),
    ...init,
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? String(payload.detail)
        : `${response.status} ${response.statusText}`;
    throw new ApiError(response.status, detail);
  }

  return payload as T;
}

export function createClientSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  return `smear-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}

export function loadOrCreateSessionId(
  storageKey = DEFAULT_SESSION_STORAGE_KEY,
): string {
  if (typeof window === "undefined") {
    return createClientSessionId();
  }

  const existing = window.localStorage.getItem(storageKey)?.trim();
  if (existing) {
    return existing;
  }

  const created = createClientSessionId();
  window.localStorage.setItem(storageKey, created);
  return created;
}

export function createApiClient({
  apiBaseUrl = DEFAULT_API_BASE_URL,
  sessionId,
}: ApiClientOptions = {}) {
  return {
    fetchBots(): Promise<ReadyBotListResponse> {
      return request<ReadyBotListResponse>(apiBaseUrl, sessionId, "/bots");
    },

    createGame(payload: NewGamePayload): Promise<GameState> {
      return request<GameState>(apiBaseUrl, sessionId, "/game/new", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },

    createLobby(payload: CreateLobbyPayload): Promise<LobbyState> {
      return request<LobbyState>(apiBaseUrl, null, "/lobbies", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },

    fetchLobby(
      lobbyCode: string,
      playerToken?: string | null,
    ): Promise<LobbyState> {
      const searchParams = new URLSearchParams();
      if (playerToken) {
        searchParams.set("player_token", playerToken);
      }
      const query = searchParams.toString();
      return request<LobbyState>(
        apiBaseUrl,
        null,
        `/lobbies/${encodeURIComponent(lobbyCode)}${query ? `?${query}` : ""}`,
      );
    },

    joinLobby(
      lobbyCode: string,
      payload: JoinLobbyPayload,
    ): Promise<LobbyState> {
      return request<LobbyState>(
        apiBaseUrl,
        null,
        `/lobbies/${encodeURIComponent(lobbyCode)}/join`,
        {
          method: "POST",
          body: JSON.stringify(payload),
        },
      );
    },

    startLobby(lobbyCode: string, playerToken: string): Promise<LobbyState> {
      return request<LobbyState>(
        apiBaseUrl,
        null,
        `/lobbies/${encodeURIComponent(lobbyCode)}/start`,
        {
          method: "POST",
          body: JSON.stringify({ player_token: playerToken }),
        },
      );
    },

    placeLobbyBid(
      lobbyCode: string,
      playerToken: string,
      amount: number,
    ): Promise<LobbyState> {
      return request<LobbyState>(
        apiBaseUrl,
        null,
        `/lobbies/${encodeURIComponent(lobbyCode)}/auction/bid`,
        {
          method: "POST",
          body: JSON.stringify({
            player_token: playerToken,
            amount,
          }),
        },
      );
    },

    passLobbyAuction(
      lobbyCode: string,
      playerToken: string,
    ): Promise<LobbyState> {
      return request<LobbyState>(
        apiBaseUrl,
        null,
        `/lobbies/${encodeURIComponent(lobbyCode)}/auction/pass`,
        {
          method: "POST",
          body: JSON.stringify({ player_token: playerToken }),
        },
      );
    },

    playLobbyCard(
      lobbyCode: string,
      playerToken: string,
      cardCode: string,
    ): Promise<LobbyState> {
      return request<LobbyState>(
        apiBaseUrl,
        null,
        `/lobbies/${encodeURIComponent(lobbyCode)}/play`,
        {
          method: "POST",
          body: JSON.stringify({
            player_token: playerToken,
            card_code: cardCode,
          }),
        },
      );
    },

    nextLobbyRound(
      lobbyCode: string,
      playerToken: string,
    ): Promise<LobbyState> {
      return request<LobbyState>(
        apiBaseUrl,
        null,
        `/lobbies/${encodeURIComponent(lobbyCode)}/next-round`,
        {
          method: "POST",
          body: JSON.stringify({ player_token: playerToken }),
        },
      );
    },

    resetRound(options?: MutatingRequestOptions): Promise<GameState> {
      return request<GameState>(
        apiBaseUrl,
        sessionId,
        withAutoRunBots("/game/reset", options),
        {
          method: "POST",
        },
      );
    },

    nextRound(options?: MutatingRequestOptions): Promise<GameState> {
      return request<GameState>(
        apiBaseUrl,
        sessionId,
        withAutoRunBots("/game/next-round", options),
        {
          method: "POST",
        },
      );
    },

    placeBid(
      amount: number,
      options?: MutatingRequestOptions,
    ): Promise<GameState> {
      return request<GameState>(apiBaseUrl, sessionId, "/game/auction/bid", {
        method: "POST",
        body: JSON.stringify({
          amount,
          auto_run_bots: options?.auto_run_bots,
        }),
      });
    },

    passAuction(options?: MutatingRequestOptions): Promise<GameState> {
      return request<GameState>(
        apiBaseUrl,
        sessionId,
        withAutoRunBots("/game/auction/pass", options),
        {
          method: "POST",
        },
      );
    },

    fetchGameState(): Promise<GameState> {
      return request<GameState>(apiBaseUrl, sessionId, "/game/state");
    },

    fetchLegalActions(): Promise<LegalActionsResponse> {
      return request<LegalActionsResponse>(
        apiBaseUrl,
        sessionId,
        "/game/legal-actions",
      );
    },

    fetchLearnChallenge(
      phase?: "auction" | "play",
      botId?: string,
    ): Promise<LearnChallenge> {
      const searchParams = new URLSearchParams();
      if (phase) {
        searchParams.set("phase", phase);
      }
      if (botId) {
        searchParams.set("bot_id", botId);
      }
      const query = searchParams.toString();
      const path = query ? `/learn/challenge?${query}` : "/learn/challenge";
      return request<LearnChallenge>(apiBaseUrl, sessionId, path);
    },

    createDonationCheckoutSession(
      payload: DonationCheckoutPayload,
    ): Promise<DonationCheckoutResponse> {
      return request<DonationCheckoutResponse>(
        apiBaseUrl,
        sessionId,
        "/donations/checkout-session",
        {
          method: "POST",
          body: JSON.stringify(payload),
        },
      );
    },

    playCard(
      cardCode: string,
      options?: MutatingRequestOptions,
    ): Promise<GameState> {
      return request<GameState>(apiBaseUrl, sessionId, "/game/play", {
        method: "POST",
        body: JSON.stringify({
          card_code: cardCode,
          auto_run_bots: options?.auto_run_bots,
        }),
      });
    },

    fetchScore(): Promise<Score> {
      return request<Score>(apiBaseUrl, sessionId, "/game/score");
    },

    async fetchOptionalScore(): Promise<Score | null> {
      try {
        return await this.fetchScore();
      } catch (error) {
        if (error instanceof ApiError && error.status === 409) {
          return null;
        }
        throw error;
      }
    },

    stepBotTurn(): Promise<GameState> {
      return request<GameState>(apiBaseUrl, sessionId, "/game/bots/step", {
        method: "POST",
      });
    },

    fetchBotProgress(): Promise<BotProgress> {
      return request<BotProgress>(
        apiBaseUrl,
        sessionId,
        "/game/bots/progress",
      );
    },

    openGameEvents(): WebSocket | null {
      if (typeof WebSocket === "undefined") {
        return null;
      }
      return new WebSocket(buildGameEventsUrl(apiBaseUrl, sessionId));
    },

    openLobbyEvents(lobbyCode: string, playerToken: string): WebSocket | null {
      if (typeof WebSocket === "undefined") {
        return null;
      }
      return new WebSocket(
        buildLobbyEventsUrl(apiBaseUrl, lobbyCode, playerToken),
      );
    },
  };
}
