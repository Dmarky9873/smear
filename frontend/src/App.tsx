import React, { useEffect, useMemo, useRef, useState } from "react";

import {
  createGame,
  fetchBots,
  fetchGameState,
  fetchLegalActions,
  fetchScore,
  nextRound,
  passAuction,
  placeBid,
  playCard,
  resetRound,
  stepBotTurn,
} from "./api";
import { DebugModeView } from "./components/DebugModeView";
import { PlayModeView } from "./components/PlayModeView";
import type {
  BidAction,
  GameState,
  LegalAction,
  PlayCardAction,
  Player,
  ReadyBot,
  Score,
} from "./types";

type AppMode = "play" | "debug";

const BOT_ACTION_DELAY_MS = 700;

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

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

export default function App() {
  const [mode, setMode] = useState<AppMode>("play");
  const [numPlayers, setNumPlayers] = useState(4);
  const [playerNames, setPlayerNames] = useState<string[]>(() =>
    buildDefaultPlayerNames(4),
  );
  const [playerBots, setPlayerBots] = useState<(string | null)[]>(() =>
    Array.from({ length: 4 }, () => null),
  );
  const [teamsInput, setTeamsInput] = useState("");
  const [availableBots, setAvailableBots] = useState<ReadyBot[]>([]);
  const [state, setState] = useState<GameState | null>(null);
  const [legalActions, setLegalActions] = useState<LegalAction[]>([]);
  const [score, setScore] = useState<Score | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [botThinkingName, setBotThinkingName] = useState<string | null>(null);
  const [botActionDelayMs, setBotActionDelayMs] = useState(BOT_ACTION_DELAY_MS);
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
    const resolvedState = nextState ?? (await fetchGameState());
    const shouldLoadScore =
      resolvedState.phase === "round_complete" ||
      resolvedState.phase === "match_complete";

    const [nextLegalActions, nextScore] = await Promise.all([
      fetchLegalActions(),
      shouldLoadScore ? fetchScore() : Promise.resolve(null),
    ]);

    setState(resolvedState);
    setLegalActions(nextLegalActions.actions);
    setScore(nextScore);
    return resolvedState;
  }

  async function loadGameState() {
    return syncStateSnapshot();
  }

  function cancelBotSequence() {
    botSequenceRef.current += 1;
    setBotThinkingName(null);
    setIsLoading(false);
  }

  async function advanceBotsForPlayMode(
    initialState: GameState,
    sequenceId: number,
  ) {
    let currentState = initialState;

    while (sequenceId === botSequenceRef.current && isBotTurn(currentState)) {
      const playerName = getTurnPlayerName(currentState);
      if (!playerName) {
        break;
      }

      setBotThinkingName(playerName);
      await delay(botActionDelayMs);

      if (sequenceId !== botSequenceRef.current) {
        return;
      }

      const nextState = await syncStateSnapshot(await stepBotTurn());
      const shouldPause =
        shouldPauseAfterBotStep(currentState, nextState) ||
        !isBotTurn(nextState);

      currentState = nextState;

      if (shouldPause) {
        setBotThinkingName(null);
        await delay(botActionDelayMs);

        if (sequenceId !== botSequenceRef.current) {
          return;
        }
      }
    }

    if (sequenceId === botSequenceRef.current) {
      setBotThinkingName(null);
    }
  }

  async function runWithErrorHandling(task: () => Promise<void>) {
    setIsLoading(true);
    setError(null);
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
    const sequenceId = botSequenceRef.current + 1;
    botSequenceRef.current = sequenceId;
    setIsLoading(true);
    setError(null);

    try {
      const nextState = await syncStateSnapshot(await task());
      await advanceBotsForPlayMode(nextState, sequenceId);
    } catch (taskError) {
      setError(
        taskError instanceof Error ? taskError.message : "Unknown error",
      );
    } finally {
      if (botSequenceRef.current === sequenceId) {
        setBotThinkingName(null);
        setIsLoading(false);
      }
    }
  }

  useEffect(() => {
    void runWithErrorHandling(async () => {
      await loadGameState();
    });
  }, []);

  useEffect(() => {
    let isActive = true;

    void fetchBots()
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
  }, []);

  useEffect(() => {
    if (mode !== "play") {
      cancelBotSequence();
    }
  }, [mode]);

  useEffect(() => {
    if (mode !== "play" || !state || isLoading || !isBotTurn(state)) {
      return;
    }

    void runPlayModeTask(async () => state);
  }, [botActionDelayMs, isLoading, mode, state]);

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
      await createGame({
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
      createGame({
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
      await resetRound();
      await loadGameState();
    });
  }

  async function handleNextRoundDebug() {
    await runWithErrorHandling(async () => {
      await nextRound();
      await loadGameState();
    });
  }

  async function handleNextRoundPlay() {
    await runPlayModeTask(async () =>
      nextRound({
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
      await playCard(cardCode);
      await loadGameState();
    });
  }

  async function handlePlayPlay(cardCode: string) {
    await runPlayModeTask(async () =>
      playCard(cardCode, {
        auto_run_bots: false,
      }),
    );
  }

  async function handleBidDebug(amount: number) {
    await runWithErrorHandling(async () => {
      await placeBid(amount);
      await loadGameState();
    });
  }

  async function handleBidPlay(amount: number) {
    await runPlayModeTask(async () =>
      placeBid(amount, {
        auto_run_bots: false,
      }),
    );
  }

  async function handlePassAuctionDebug() {
    await runWithErrorHandling(async () => {
      await passAuction();
      await loadGameState();
    });
  }

  async function handlePassAuctionPlay() {
    await runPlayModeTask(async () =>
      passAuction({
        auto_run_bots: false,
      }),
    );
  }

  const currentTurnName = getTurnPlayerName(state) ?? "No game";

  return (
    <>
      <header className="mode-switcher">
        <div className="mode-switcher__copy">
          <strong>Frontend mode</strong>
          <span>Play is table-first. Debug keeps the full state inspector.</span>
        </div>
        <div className="mode-switcher__controls">
          <button
            type="button"
            className={mode === "play" ? "mode-switcher__button is-active" : "mode-switcher__button"}
            onClick={() => setMode("play")}
          >
            Play
          </button>
          <button
            type="button"
            className={mode === "debug" ? "mode-switcher__button is-active" : "mode-switcher__button"}
            onClick={() => setMode("debug")}
          >
            Debug
          </button>
        </div>
      </header>

      {mode === "debug" ? (
        <DebugModeView
          numPlayers={numPlayers}
          playerNames={playerNames}
          playerBots={playerBots}
          teamsInput={teamsInput}
          availableBots={availableBots}
          state={state}
          score={score}
          error={error}
          isLoading={isLoading}
          currentTurnName={currentTurnName}
          currentPlayer={currentPlayer}
          playActions={playActions}
          bidActions={bidActions}
          canPassAuction={canPassAuction}
          onNumPlayersChange={setNumPlayers}
          onPlayerNameChange={handlePlayerNameChange}
          onPlayerBotChange={handlePlayerBotChange}
          onTeamsInputChange={setTeamsInput}
          onNewGame={() => {
            void handleNewGameDebug();
          }}
          onResetRound={() => {
            void handleResetRoundDebug();
          }}
          onNextRound={() => {
            void handleNextRoundDebug();
          }}
          onRefreshState={() => {
            void handleRefreshState();
          }}
          onPlay={handlePlayDebug}
          onBid={(amount) => {
            void handleBidDebug(amount);
          }}
          onPassAuction={() => {
            void handlePassAuctionDebug();
          }}
        />
      ) : (
        <PlayModeView
          numPlayers={numPlayers}
          playerNames={playerNames}
          playerBots={playerBots}
          teamsInput={teamsInput}
          availableBots={availableBots}
          state={state}
          score={score}
          error={error}
          isBusy={isLoading}
          botThinkingName={botThinkingName}
          botActionDelayMs={botActionDelayMs}
          currentTurnName={currentTurnName}
          turnPlayer={turnPlayer}
          playActions={playActions}
          bidActions={bidActions}
          canPassAuction={canPassAuction}
          onNumPlayersChange={setNumPlayers}
          onPlayerNameChange={handlePlayerNameChange}
          onPlayerBotChange={handlePlayerBotChange}
          onTeamsInputChange={setTeamsInput}
          onBotActionDelayChange={setBotActionDelayMs}
          onNewGame={() => {
            void handleNewGamePlay();
          }}
          onNextRound={() => {
            void handleNextRoundPlay();
          }}
          onPlay={handlePlayPlay}
          onBid={(amount) => {
            void handleBidPlay(amount);
          }}
          onPassAuction={() => {
            void handlePassAuctionPlay();
          }}
        />
      )}
    </>
  );
}
