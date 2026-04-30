import { useEffect, useMemo, useState } from "react";

import {
  createGame,
  fetchGameState,
  fetchLegalActions,
  fetchScore,
  playCard,
  resetRound,
} from "./api";
import { CurrentTrickPanel } from "./components/CurrentTrickPanel";
import { DebugJsonPanel } from "./components/DebugJsonPanel";
import { LegalActionsPanel } from "./components/LegalActionsPanel";
import { PlayingCard } from "./components/PlayingCard";
import { PlayerPanel } from "./components/PlayerPanel";
import type { GameState, Score } from "./types";

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

export default function App() {
  const [numPlayers, setNumPlayers] = useState(4);
  const [playerNames, setPlayerNames] = useState<string[]>(() =>
    buildDefaultPlayerNames(4),
  );
  const [teamsInput, setTeamsInput] = useState("");
  const [state, setState] = useState<GameState | null>(null);
  const [legalActions, setLegalActions] = useState<string[]>([]);
  const [score, setScore] = useState<Score | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setPlayerNames((current) => {
      const next = buildDefaultPlayerNames(numPlayers);
      return next.map((fallbackName, index) => current[index] ?? fallbackName);
    });
  }, [numPlayers]);

  const currentPlayer = useMemo(() => {
    if (!state) {
      return null;
    }

    return (
      state.round.players.find(
        (player) => player.name === state.round.current_player_name,
      ) ?? null
    );
  }, [state]);

  async function loadGameState() {
    const nextState = await fetchGameState();
    const nextLegalActions = await fetchLegalActions();
    setState(nextState);
    setLegalActions(nextLegalActions.actions.map((action) => action.card_code));

    if (nextState.round.is_terminal) {
      setScore(await fetchScore());
    } else {
      setScore(null);
    }
  }

  async function runWithErrorHandling(task: () => Promise<void>) {
    setIsLoading(true);
    setError(null);
    try {
      await task();
    } catch (taskError) {
      setError(taskError instanceof Error ? taskError.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void runWithErrorHandling(loadGameState);
  }, []);

  async function handleNewGame() {
    await runWithErrorHandling(async () => {
      const normalizedNames = playerNames.map((name, index) => {
        const trimmed = name.trim();
        return trimmed || `Player ${index + 1}`;
      });

      await createGame({
        num_players: numPlayers,
        player_names: normalizedNames,
        teams: parseTeams(teamsInput),
      });

      await loadGameState();
    });
  }

  async function handleResetRound() {
    await runWithErrorHandling(async () => {
      await resetRound();
      await loadGameState();
    });
  }

  async function handleRefreshState() {
    await runWithErrorHandling(loadGameState);
  }

  async function handlePlay(cardCode: string) {
    await runWithErrorHandling(async () => {
      await playCard(cardCode);
      await loadGameState();
    });
  }

  return (
    <main className="app-shell">
      <header className="status-bar panel">
        <div>
          <strong>Current player:</strong>{" "}
          {state?.round.current_player_name ?? "No game"}
        </div>
        <div>
          <strong>Trump:</strong> {state?.round.trump ?? "Unset"}
        </div>
        <div>
          <strong>Terminal:</strong>{" "}
          {state?.round.is_terminal ? "Yes" : "No"}
        </div>
        <div>
          <strong>Current trick leader:</strong>{" "}
          {state?.round.current_trick.leader_name ?? "N/A"}
        </div>
        <div>
          <strong>Completed tricks:</strong>{" "}
          {state?.round.trick_history.length ?? 0}
        </div>
        <div>
          <strong>Hidden cards:</strong>{" "}
          {state?.round.hidden_cards_count ?? 0}
        </div>
        <div>
          <strong>Deck low:</strong> {state?.low ?? "N/A"}
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="panel">
        <h2>Setup</h2>
        <div className="setup-grid">
          <label>
            Number of players
            <input
              type="number"
              min={3}
              max={8}
              value={numPlayers}
              onChange={(event) =>
                setNumPlayers(Number(event.target.value) || 3)
              }
            />
          </label>

          {playerNames.map((playerName, index) => (
            <label key={index}>
              Player {index + 1}
              <input
                type="text"
                value={playerName}
                onChange={(event) => {
                  const nextNames = [...playerNames];
                  nextNames[index] = event.target.value;
                  setPlayerNames(nextNames);
                }}
              />
            </label>
          ))}

          <label className="setup-grid__full">
            Teams (optional, one team per line, comma separated)
            <textarea
              rows={4}
              value={teamsInput}
              onChange={(event) => setTeamsInput(event.target.value)}
              placeholder={"Player 1, Player 3\nPlayer 2, Player 4"}
            />
          </label>
        </div>

        <div className="button-row">
          <button type="button" onClick={handleNewGame} disabled={isLoading}>
            New Game
          </button>
          <button
            type="button"
            onClick={handleResetRound}
            disabled={isLoading || !state}
          >
            Reset Round
          </button>
          <button type="button" onClick={handleRefreshState} disabled={isLoading}>
            Refresh State
          </button>
        </div>
      </section>

      {state ? (
        <>
          <section className="panel">
            <h2>Players</h2>
            <div className="players-grid">
              {state.round.players.map((player) => (
                <PlayerPanel
                  key={player.name}
                  player={player}
                  trump={state.round.trump}
                  isCurrentPlayer={player.name === state.round.current_player_name}
                />
              ))}
            </div>
          </section>

          <CurrentTrickPanel trick={state.round.current_trick} />

          <LegalActionsPanel
            actions={legalActions.map((cardCode) => ({
              type: "play_card" as const,
              card_code: cardCode,
            }))}
            currentPlayer={currentPlayer}
            trump={state.round.trump}
            isTerminal={state.round.is_terminal}
            disabled={isLoading}
            onPlay={handlePlay}
          />

          <section className="panel">
            <h2>Trick History</h2>
            <div className="list-block">
              {state.round.trick_history.length === 0 ? (
                <span className="muted">No completed tricks yet.</span>
              ) : (
                state.round.trick_history.map((trick, index) => (
                  <div key={`${trick.leader_name}-${index}`} className="history-block">
                    <div className="history-block__header">
                      <strong>Trick {index + 1}</strong>
                      <span>
                        Leader: {trick.leader_name} | Winner:{" "}
                        {trick.winner_name ?? "Unknown"}
                      </span>
                    </div>
                    <div className="card-row">
                      {trick.plays.map((play, playIndex) => (
                        <div
                          key={`${play.player_name}-${play.card.code}-${playIndex}`}
                          className="history-play"
                        >
                          <span>{play.player_name}</span>
                          <PlayingCard card={play.card} compact />
                        </div>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

          <section className="panel">
            <h2>Score</h2>
            {!state.round.is_terminal ? (
              <p className="muted">Score is only available once the round is terminal.</p>
            ) : score ? (
              <div className="score-grid">
                {Object.entries(score).map(([name, value]) => (
                  <div key={name} className="score-cell">
                    <strong>{name}</strong>
                    <span>{value}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">No score available.</p>
            )}
          </section>

          <section className="panel">
            <h2>Teams</h2>
            <div className="list-block">
              {state.round.teams.map((team, index) => (
                <div key={index} className="list-row list-row--stacked">
                  <span>
                    <strong>Team {index + 1}:</strong> {team.constituents.join(", ")}
                  </span>
                  <span>Captured cards: {team.captured_count}</span>
                </div>
              ))}
            </div>
          </section>

          <DebugJsonPanel value={state} />
        </>
      ) : null}
    </main>
  );
}
