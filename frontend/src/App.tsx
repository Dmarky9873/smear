import { useEffect, useState } from "react";
import type { FormEvent } from "react";

import {
  API_BASE_URL,
  applyGameAction,
  createNewGame,
  getGameDebug,
  getGameState,
  getLegalActions,
  resetGame,
  runAiMove,
} from "./api";
import { DebugJsonPanel } from "./components/DebugJsonPanel";
import { LegalActionsPanel } from "./components/LegalActionsPanel";
import { PlayerPanel } from "./components/PlayerPanel";
import { TablePanel } from "./components/TablePanel";
import { TopStatusBar } from "./components/TopStatusBar";
import { defaultNewGameRequest } from "./mockState";
import type { GameDebugResponse, GameState, LegalAction } from "./types";

type CreateGameFormState = {
  playerCount: number;
  playerNames: string;
  seed: string;
  debug: boolean;
};

function App() {
  const [gameId, setGameId] = useState<string | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [debugPayload, setDebugPayload] = useState<GameDebugResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeRequest, setActiveRequest] = useState<string | null>(null);
  const [createForm, setCreateForm] = useState<CreateGameFormState>({
    playerCount: defaultNewGameRequest.player_count ?? 4,
    playerNames: (defaultNewGameRequest.player_names ?? []).join(", "),
    seed: "",
    debug: defaultNewGameRequest.debug ?? true,
  });

  async function loadGame(nextGameId: string) {
    setLoading(true);
    setError(null);

    try {
      const [state, legalActions, debug] = await Promise.all([
        getGameState(nextGameId),
        getLegalActions(nextGameId),
        getGameDebug(nextGameId),
      ]);

      setGameState({
        ...state,
        legal_actions: legalActions.legal_actions,
      });
      setDebugPayload(debug);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Failed to load game state.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!gameId) {
      return;
    }

    void loadGame(gameId);
  }, [gameId]);

  async function handleCreateGame(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setActiveRequest("create");
    setError(null);

    try {
      const playerNames = createForm.playerNames
        .split(",")
        .map((name) => name.trim())
        .filter(Boolean);

      const state = await createNewGame({
        player_count: createForm.playerCount,
        player_names: playerNames.length > 0 ? playerNames : undefined,
        seed: createForm.seed ? Number(createForm.seed) : undefined,
        debug: createForm.debug,
      });

      setGameId(state.game_id);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Failed to create game.");
    } finally {
      setActiveRequest(null);
    }
  }

  async function runMutation(
    label: string,
    callback: () => Promise<unknown>,
    reloadAfter = true,
  ) {
    setActiveRequest(label);
    setError(null);

    try {
      await callback();
      if (reloadAfter && gameId) {
        await loadGame(gameId);
      }
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Request failed.");
    } finally {
      setActiveRequest(null);
    }
  }

  async function handleAction(action: LegalAction) {
    if (!gameId) {
      return;
    }

    await runMutation(`action:${action.type}`, () => applyGameAction(gameId, action));
  }

  async function handleRefresh() {
    if (!gameId) {
      return;
    }

    await runMutation("refresh", () => loadGame(gameId), false);
  }

  async function handleReset() {
    if (!gameId) {
      return;
    }

    await runMutation("reset", () => resetGame(gameId));
  }

  async function handleAiMove() {
    if (!gameId) {
      return;
    }

    await runMutation("ai", () => runAiMove(gameId));
  }

  const isBusy = loading || activeRequest !== null;

  if (!gameState || !gameId) {
    return (
      <main className="app-shell">
        <section className="panel create-panel">
          <header className="panel-header">
            <div>
              <h1>Create New Game</h1>
              <p>Start a local mock Smear game and inspect the backend state flow.</p>
            </div>
          </header>

          <form className="create-form" onSubmit={handleCreateGame}>
            <label>
              Player Count
              <input
                max={5}
                min={2}
                onChange={(event) =>
                  setCreateForm((current) => ({
                    ...current,
                    playerCount: Number(event.target.value),
                  }))
                }
                type="number"
                value={createForm.playerCount}
              />
            </label>

            <label>
              Player Names
              <input
                onChange={(event) =>
                  setCreateForm((current) => ({
                    ...current,
                    playerNames: event.target.value,
                  }))
                }
                placeholder="North, East, South, West"
                type="text"
                value={createForm.playerNames}
              />
            </label>

            <label>
              Seed
              <input
                onChange={(event) =>
                  setCreateForm((current) => ({
                    ...current,
                    seed: event.target.value,
                  }))
                }
                placeholder="Optional"
                type="number"
                value={createForm.seed}
              />
            </label>

            <label className="checkbox-row">
              <input
                checked={createForm.debug}
                onChange={(event) =>
                  setCreateForm((current) => ({
                    ...current,
                    debug: event.target.checked,
                  }))
                }
                type="checkbox"
              />
              Enable debug metadata
            </label>

            <div className="button-row">
              <button className="primary-button" disabled={isBusy} type="submit">
                {activeRequest === "create" ? "Creating..." : "Create New Game"}
              </button>
            </div>
          </form>

          <p className="muted">Backend URL: {API_BASE_URL}</p>
          {error ? <p className="error-text">{error}</p> : null}
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <TopStatusBar state={gameState} />

      <section className="panel">
        <div className="button-row">
          <button className="primary-button" disabled={isBusy} onClick={() => void handleRefresh()} type="button">
            {activeRequest === "refresh" || loading ? "Refreshing..." : "Refresh"}
          </button>
          <button className="secondary-button" disabled={isBusy} onClick={() => void handleReset()} type="button">
            {activeRequest === "reset" ? "Resetting..." : "Reset"}
          </button>
          <button className="secondary-button" disabled={isBusy} onClick={() => void handleAiMove()} type="button">
            {activeRequest === "ai" ? "Running AI..." : "AI Move"}
          </button>
          <button
            className="ghost-button"
            disabled={isBusy}
            onClick={() => {
              setGameId(null);
              setGameState(null);
              setDebugPayload(null);
              setError(null);
            }}
            type="button"
          >
            New Session
          </button>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="players-grid">
        {gameState.players.map((player) => (
          <PlayerPanel
            key={player.id}
            isCurrent={player.id === gameState.current_player_id}
            player={player}
            trumpSuit={gameState.trump_suit}
          />
        ))}
      </section>

      <TablePanel state={gameState} />

      <section className="two-column-grid">
        <section className="panel">
          <header className="panel-header">
            <div>
              <h2>Auction History</h2>
              <p>Mock bidding events emitted by the backend.</p>
            </div>
          </header>

          {gameState.auction_history.length > 0 ? (
            <ul className="list-block">
              {gameState.auction_history.map((entry, index) => (
                <li key={`${entry.player_id}-${entry.action}-${index}`}>
                  Player {entry.player_id}: {entry.action}
                  {entry.value !== null ? ` ${entry.value}` : ""}
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No auction history yet.</p>
          )}
        </section>

        <section className="panel">
          <header className="panel-header">
            <div>
              <h2>Logs</h2>
              <p>Backend log messages for the current state transitions.</p>
            </div>
          </header>

          {gameState.logs.length > 0 ? (
            <ul className="list-block">
              {gameState.logs.map((entry, index) => (
                <li key={`${entry}-${index}`}>{entry}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">No logs yet.</p>
          )}
        </section>
      </section>

      <LegalActionsPanel
        actions={gameState.legal_actions}
        disabled={isBusy}
        onAction={(action) => {
          void handleAction(action);
        }}
      />

      <DebugJsonPanel title="Raw JSON Debug Panel" value={debugPayload ?? gameState} />
    </main>
  );
}

export default App;
