import { useEffect, useMemo, useRef, useState } from "react";

import { ApiError, createApiClient } from "../api";
import type {
  BotProgress,
  BidAction,
  GameState,
  LegalAction,
  PlayCardAction,
  Player,
  ReadyBot,
  Score,
  TrickState,
} from "../types";

const DEFAULT_BOT_ACTION_DELAY_MS = 700;

export type SmearSetupDraft = {
  numPlayers: number;
  playerNames: string[];
  playerBots: (string | null)[];
  teamsInput: string;
};

export type UseSmearGameOptions = {
  sessionId: string;
  apiBaseUrl?: string;
  botAutomationEnabled?: boolean;
  initialNumPlayers?: number;
  initialPlayerNames?: string[];
  initialPlayerBots?: (string | null)[];
  initialTeamsInput?: string;
  initialBotActionDelayMs?: number;
};

function buildDefaultPlayerNames(count: number): string[] {
  return Array.from({ length: count }, (_, index) => `Player ${index + 1}`);
}

function parseTeams(input: string): string[][] | null {
  const rows = input
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (rows.length === 0) {
    return null;
  }

  return rows.map((row) =>
    row
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean),
  );
}

function getTurnPlayerName(state: GameState | null): string | null {
  if (!state) {
    return null;
  }

  if (state.phase === "auction") {
    return state.auction.current_bidder_name;
  }

  return state.round.current_player_name;
}

function getTurnPlayer(state: GameState | null): Player | null {
  const playerName = getTurnPlayerName(state);
  if (!state || !playerName) {
    return null;
  }

  return (
    state.round.players.find((player) => player.name === playerName) ?? null
  );
}

function isBotTurn(state: GameState | null): boolean {
  if (!state || (state.phase !== "auction" && state.phase !== "play")) {
    return false;
  }

  const player = getTurnPlayer(state);
  return Boolean(player?.bot_id);
}

function shouldPauseAfterBotStep(
  previousState: GameState,
  nextState: GameState,
): boolean {
  if (
    nextState.round.trick_history.length > previousState.round.trick_history.length
  ) {
    return true;
  }

  if (previousState.phase === "auction" && nextState.phase !== "auction") {
    return true;
  }

  if (previousState.phase === "play" && nextState.phase !== "play") {
    return true;
  }

  return false;
}

function shouldWaitForNextTrick(
  previousState: GameState | null,
  nextState: GameState,
): boolean {
  if (!previousState) {
    return false;
  }

  return (
    nextState.phase === "play" &&
    nextState.round.trick_history.length > previousState.round.trick_history.length
  );
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

export function useSmearGame({
  sessionId,
  apiBaseUrl,
  botAutomationEnabled = false,
  initialNumPlayers = 4,
  initialPlayerNames,
  initialPlayerBots,
  initialTeamsInput = "",
  initialBotActionDelayMs = DEFAULT_BOT_ACTION_DELAY_MS,
}: UseSmearGameOptions) {
  const api = useMemo(
    () =>
      createApiClient({
        apiBaseUrl,
        sessionId,
      }),
    [apiBaseUrl, sessionId],
  );

  const [numPlayers, setNumPlayers] = useState(initialNumPlayers);
  const [playerNames, setPlayerNames] = useState<string[]>(
    initialPlayerNames ?? buildDefaultPlayerNames(initialNumPlayers),
  );
  const [playerBots, setPlayerBots] = useState<(string | null)[]>(
    initialPlayerBots ?? Array.from({ length: initialNumPlayers }, () => null),
  );
  const [teamsInput, setTeamsInput] = useState(initialTeamsInput);
  const [availableBots, setAvailableBots] = useState<ReadyBot[]>([]);
  const [botProgress, setBotProgress] = useState<BotProgress | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [legalActions, setLegalActions] = useState<LegalAction[]>([]);
  const [score, setScore] = useState<Score | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [botThinkingName, setBotThinkingName] = useState<string | null>(null);
  const [revealedTrick, setRevealedTrick] = useState<TrickState | null>(null);
  const [awaitingNextTrick, setAwaitingNextTrick] = useState(false);
  const [botActionDelayMs, setBotActionDelayMs] = useState(
    initialBotActionDelayMs,
  );
  const botSequenceRef = useRef(0);

  useEffect(() => {
    setPlayerNames((current) => {
      const next = buildDefaultPlayerNames(numPlayers);
      return next.map((fallbackName, index) => current[index] ?? fallbackName);
    });
  }, [numPlayers]);

  useEffect(() => {
    setPlayerBots((current) => {
      const next = Array.from({ length: numPlayers }, () => null);
      return next.map((fallbackBot, index) => current[index] ?? fallbackBot);
    });
  }, [numPlayers]);

  useEffect(() => {
    if (
      !revealedTrick ||
      !state ||
      state.phase !== "play" ||
      state.round.current_trick.plays.length === 0
    ) {
      return;
    }

    setRevealedTrick(null);
  }, [revealedTrick, state]);

  const currentPlayer = useMemo(() => {
    if (!state || state.phase === "auction") {
      return null;
    }

    return (
      state.round.players.find(
        (player) => player.name === state.round.current_player_name,
      ) ?? null
    );
  }, [state]);

  const turnPlayer = useMemo(() => getTurnPlayer(state), [state]);

  const playActions = useMemo(
    () =>
      legalActions.filter(
        (action): action is PlayCardAction => action.type === "play_card",
      ),
    [legalActions],
  );

  const bidActions = useMemo(
    () =>
      legalActions.filter((action): action is BidAction => action.type === "bid"),
    [legalActions],
  );

  const canPassAuction = useMemo(
    () => legalActions.some((action) => action.type === "pass"),
    [legalActions],
  );

  async function syncStateSnapshot(nextState?: GameState): Promise<GameState> {
    const resolvedState = nextState ?? (await api.fetchGameState());

    const [nextLegalActions, nextScore] = await Promise.all([
      api.fetchLegalActions(),
      api.fetchOptionalScore(),
    ]);

    setState(resolvedState);
    setLegalActions(nextLegalActions.actions);
    setScore(nextScore);
    return resolvedState;
  }

  async function loadGameState() {
    try {
      return await syncStateSnapshot();
    } catch (taskError) {
      if (taskError instanceof ApiError && taskError.status === 404) {
        setState(null);
        setLegalActions([]);
        setScore(null);
        return null;
      }
      throw taskError;
    }
  }

  function cancelBotSequence() {
    botSequenceRef.current += 1;
    setAwaitingNextTrick(false);
    setBotThinkingName(null);
    setBotProgress(null);
    setRevealedTrick(null);
    setIsLoading(false);
  }

  async function advanceBotsForPlayMode(
    initialState: GameState,
    sequenceId: number,
  ): Promise<boolean> {
    let currentState = initialState;

    while (sequenceId === botSequenceRef.current && isBotTurn(currentState)) {
      const playerName = getTurnPlayerName(currentState);
      if (!playerName) {
        break;
      }

      setBotThinkingName(playerName);
      await delay(botActionDelayMs);

      if (sequenceId !== botSequenceRef.current) {
        return false;
      }

      const nextState = await syncStateSnapshot(await api.stepBotTurn());
      const completedTrick =
        nextState.round.trick_history.length > currentState.round.trick_history.length
          ? nextState.round.trick_history[nextState.round.trick_history.length - 1]
          : null;

      if (completedTrick && shouldWaitForNextTrick(currentState, nextState)) {
        setRevealedTrick(completedTrick);
        setAwaitingNextTrick(true);
        setBotThinkingName(null);
        setBotProgress(null);
        return true;
      }

      const shouldPause =
        shouldPauseAfterBotStep(currentState, nextState) ||
        !isBotTurn(nextState);

      currentState = nextState;

      if (shouldPause) {
        if (completedTrick) {
          setRevealedTrick(completedTrick);
        }
        setBotThinkingName(null);
        setBotProgress(null);
        await delay(botActionDelayMs);

        if (sequenceId !== botSequenceRef.current) {
          return false;
        }

        setRevealedTrick(null);
      }
    }

    if (sequenceId === botSequenceRef.current) {
      setBotThinkingName(null);
      setBotProgress(null);
      setRevealedTrick(null);
    }

    return false;
  }

  async function runWithErrorHandling(task: () => Promise<void>) {
    setIsLoading(true);
    setError(null);
    setAwaitingNextTrick(false);
    try {
      await task();
    } catch (taskError) {
      setError(
        taskError instanceof Error ? taskError.message : "Unknown error",
      );
    } finally {
      setIsLoading(false);
    }
  }

  async function runPlayModeTask(task: () => Promise<GameState>) {
    const previousState = state;
    const sequenceId = botSequenceRef.current + 1;
    let preserveRevealedTrick = false;
    botSequenceRef.current = sequenceId;
    setIsLoading(true);
    setError(null);
    setAwaitingNextTrick(false);

    try {
      const nextState = await syncStateSnapshot(await task());
      const completedTrick =
        previousState &&
        nextState.round.trick_history.length > previousState.round.trick_history.length
          ? nextState.round.trick_history[nextState.round.trick_history.length - 1]
          : null;

      if (completedTrick && shouldWaitForNextTrick(previousState, nextState)) {
        setRevealedTrick(completedTrick);
        setBotThinkingName(null);
        setBotProgress(null);
        setAwaitingNextTrick(true);
        preserveRevealedTrick = true;
        return;
      }

      preserveRevealedTrick = await advanceBotsForPlayMode(nextState, sequenceId);
    } catch (taskError) {
      setError(
        taskError instanceof Error ? taskError.message : "Unknown error",
      );
    } finally {
      if (botSequenceRef.current === sequenceId) {
        setBotThinkingName(null);
        setBotProgress(null);
        if (!preserveRevealedTrick) {
          setRevealedTrick(null);
        }
        setIsLoading(false);
      }
    }
  }

  useEffect(() => {
    void runWithErrorHandling(async () => {
      await loadGameState();
    });
  }, [api]);

  useEffect(() => {
    let isActive = true;

    void api
      .fetchBots()
      .then((payload) => {
        if (isActive) {
          setAvailableBots(payload.bots);
        }
      })
      .catch((botError) => {
        if (isActive) {
          setError(
            botError instanceof Error ? botError.message : "Failed to load bots",
          );
        }
      });

    return () => {
      isActive = false;
    };
  }, [api]);

  useEffect(() => {
    if (botAutomationEnabled) {
      return;
    }

    cancelBotSequence();
  }, [botAutomationEnabled]);

  useEffect(() => {
    if (!botAutomationEnabled || !botThinkingName || awaitingNextTrick) {
      setBotProgress(null);
      return;
    }

    let cancelled = false;
    let timeoutId: number | null = null;

    async function pollProgress() {
      try {
        const progress = await api.fetchBotProgress();
        if (!cancelled) {
          setBotProgress(progress.active ? progress : null);
        }
      } catch {
        if (!cancelled) {
          setBotProgress(null);
        }
      } finally {
        if (!cancelled) {
          timeoutId = window.setTimeout(pollProgress, 150);
        }
      }
    }

    void pollProgress();

    return () => {
      cancelled = true;
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [api, awaitingNextTrick, botAutomationEnabled, botThinkingName]);

  useEffect(() => {
    if (
      !botAutomationEnabled ||
      !state ||
      isLoading ||
      awaitingNextTrick ||
      !isBotTurn(state)
    ) {
      return;
    }

    void runPlayModeTask(async () => state);
  }, [
    awaitingNextTrick,
    botActionDelayMs,
    botAutomationEnabled,
    isLoading,
    state,
  ]);

  function normalizePlayerNames(): string[] {
    return playerNames.map((name, index) => {
      const trimmed = name.trim();
      return trimmed || `Player ${index + 1}`;
    });
  }

  function handlePlayerNameChange(index: number, value: string) {
    setPlayerNames((current) => {
      const next = [...current];
      next[index] = value;
      return next;
    });
  }

  function handlePlayerBotChange(index: number, value: string | null) {
    setPlayerBots((current) => {
      const next = [...current];
      next[index] = value;
      return next;
    });
  }

  async function handleNewGameDebug() {
    await runWithErrorHandling(async () => {
      await api.createGame({
        num_players: numPlayers,
        player_names: normalizePlayerNames(),
        teams: parseTeams(teamsInput),
        player_bots: playerBots,
      });

      await loadGameState();
    });
  }

  async function handleNewGamePlay() {
    await runPlayModeTask(async () =>
      api.createGame({
        num_players: numPlayers,
        player_names: normalizePlayerNames(),
        teams: parseTeams(teamsInput),
        player_bots: playerBots,
        auto_run_bots: false,
      }),
    );
  }

  async function handleResetRoundDebug() {
    await runWithErrorHandling(async () => {
      await api.resetRound();
      await loadGameState();
    });
  }

  async function handleNextRoundDebug() {
    await runWithErrorHandling(async () => {
      await api.nextRound();
      await loadGameState();
    });
  }

  async function handleNextRoundPlay() {
    await runPlayModeTask(async () =>
      api.nextRound({
        auto_run_bots: false,
      }),
    );
  }

  async function handleRefreshState() {
    await runWithErrorHandling(async () => {
      await loadGameState();
    });
  }

  async function handlePlayDebug(cardCode: string) {
    await runWithErrorHandling(async () => {
      await api.playCard(cardCode);
      await loadGameState();
    });
  }

  async function handlePlayPlay(cardCode: string) {
    await runPlayModeTask(async () =>
      api.playCard(cardCode, {
        auto_run_bots: false,
      }),
    );
  }

  async function handleBidDebug(amount: number) {
    await runWithErrorHandling(async () => {
      await api.placeBid(amount);
      await loadGameState();
    });
  }

  async function handleBidPlay(amount: number) {
    await runPlayModeTask(async () =>
      api.placeBid(amount, {
        auto_run_bots: false,
      }),
    );
  }

  async function handlePassAuctionDebug() {
    await runWithErrorHandling(async () => {
      await api.passAuction();
      await loadGameState();
    });
  }

  async function handlePassAuctionPlay() {
    await runPlayModeTask(async () =>
      api.passAuction({
        auto_run_bots: false,
      }),
    );
  }

  function handleAdvanceToNextTrick() {
    setRevealedTrick(null);
    setAwaitingNextTrick(false);
  }

  function applySetupDraft(draft: SmearSetupDraft) {
    setNumPlayers(draft.numPlayers);
    setPlayerNames(draft.playerNames);
    setPlayerBots(draft.playerBots);
    setTeamsInput(draft.teamsInput);
  }

  return {
    availableBots,
    awaitingNextTrick,
    bidActions,
    botActionDelayMs,
    botProgress,
    botThinkingName,
    canPassAuction,
    currentPlayer,
    currentTurnName: getTurnPlayerName(state) ?? "No game",
    error,
    isLoading,
    legalActions,
    numPlayers,
    playActions,
    playerBots,
    playerNames,
    revealedTrick,
    score,
    state,
    teamsInput,
    turnPlayer,
    applySetupDraft,
    handleAdvanceToNextTrick,
    handleBidDebug,
    handleBidPlay,
    handleNewGameDebug,
    handleNewGamePlay,
    handleNextRoundDebug,
    handleNextRoundPlay,
    handlePassAuctionDebug,
    handlePassAuctionPlay,
    handlePlayDebug,
    handlePlayPlay,
    handlePlayerBotChange,
    handlePlayerNameChange,
    handleRefreshState,
    handleResetRoundDebug,
    setBotActionDelayMs,
    setNumPlayers,
    setTeamsInput,
  };
}
