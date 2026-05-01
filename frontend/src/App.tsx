import React, { useEffect, useMemo, useState } from "react";

import {
  createGame,
  fetchGameState,
  fetchLegalActions,
  nextRound,
  fetchScore,
  passAuction,
  placeBid,
  playCard,
  resetRound,
} from "./api";
import { CurrentTrickPanel } from "./components/CurrentTrickPanel";
import { DebugJsonPanel } from "./components/DebugJsonPanel";
import { LegalActionsPanel } from "./components/LegalActionsPanel";
import { PlayingCard } from "./components/PlayingCard";
import { PlayerPanel } from "./components/PlayerPanel";
import type { BidAction, GameState, LegalAction, PlayCardAction, Score } from "./types";

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
  const [legalActions, setLegalActions] = useState<LegalAction[]>([]);
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
    if (!state || state.phase === "auction") {
      return null;
    }

    return (
      state.round.players.find(
        (player) => player.name === state.round.current_player_name,
      ) ?? null
    );
  }, [state]);

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

  async function loadGameState() {
    const nextState = await fetchGameState();
    const nextLegalActions = await fetchLegalActions();
    setState(nextState);
    setLegalActions(nextLegalActions.actions);

    if (
      nextState.phase === "round_complete" ||
      nextState.phase === "match_complete"
    ) {
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
      setError(
        taskError instanceof Error ? taskError.message : "Unknown error",
      );
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

  async function handleNextRound() {
    await runWithErrorHandling(async () => {
      await nextRound();
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

  async function handleBid(amount: number) {
    await runWithErrorHandling(async () => {
      await placeBid(amount);
      await loadGameState();
    });
  }

  async function handlePassAuction() {
    await runWithErrorHandling(async () => {
      await passAuction();
      await loadGameState();
    });
  }

  const currentTurnName = state
    ? state.phase === "auction"
      ? state.auction.current_bidder_name
      : state.round.current_player_name
    : "No game";

  return (
    <main className="app-shell">
      <header className="status-bar panel">
        <div>
          <strong>Phase:</strong> {state?.phase ?? "No game"}
        </div>
        <div>
          <strong>Round:</strong> {state?.match.round_number ?? "N/A"}
        </div>
        <div>
          <strong>Current turn:</strong> {currentTurnName}
        </div>
        <div>
          <strong>Trump:</strong> {state?.round.trump ?? "Unset"}
        </div>
        <div>
          <strong>Highest bid:</strong>{" "}
          {state?.auction.current_high_bid ?? "No bid yet"}
        </div>
        <div>
          <strong>High bidder:</strong>{" "}
          {state?.auction.highest_bidder_name ?? "None"}
        </div>
        <div>
          <strong>Dealer:</strong> {state?.auction.dealer_name ?? "N/A"}
        </div>
        <div>
          <strong>Hidden cards:</strong> {state?.round.hidden_cards_count ?? 0}
        </div>
        <div>
          <strong>Completed tricks:</strong>{" "}
          {state?.round.trick_history.length ?? 0}
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
            Reset Round (Debug)
          </button>
          <button
            type="button"
            onClick={handleNextRound}
            disabled={
              isLoading ||
              !state ||
              (state.phase !== "round_complete" && state.phase !== "match_complete") ||
              state.match.is_complete
            }
          >
            Next Round
          </button>
          <button
            type="button"
            onClick={handleRefreshState}
            disabled={isLoading}
          >
            Refresh State
          </button>
        </div>
      </section>

      {state ? (
        <>
          <section className="panel">
            <h2>Match Score</h2>
            <div className="score-awards">
              <div className="score-award-card">
                <strong>Target</strong>
                <span>{state.match.target_score} points</span>
              </div>
              <div className="score-award-card">
                <strong>Round</strong>
                <span>{state.match.round_number}</span>
              </div>
              <div className="score-award-card">
                <strong>Status</strong>
                <span>
                  {state.match.is_complete
                    ? `Winner: ${state.match.winner_names.join(", ")}`
                    : "Match in progress"}
                </span>
              </div>
            </div>

            <div className="score-grid">
              {state.match.scores.map((entry) => (
                <div key={entry.name} className="score-cell">
                  <strong>{entry.name}</strong>
                  <span>{entry.points}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <h2>Auction</h2>
            <div className="auction-summary">
              <div className="score-award-card">
                <strong>Dealer</strong>
                <span>{state.auction.dealer_name}</span>
              </div>
              <div className="score-award-card">
                <strong>Current bidder</strong>
                <span>{state.auction.current_bidder_name}</span>
              </div>
              <div className="score-award-card">
                <strong>Current high bid</strong>
                <span>
                  {state.auction.current_high_bid !== null
                    ? state.auction.current_high_bid
                    : "No bid yet"}
                </span>
              </div>
              <div className="score-award-card">
                <strong>High bidder</strong>
                <span>{state.auction.highest_bidder_name ?? "None"}</span>
              </div>
            </div>

            <div className="list-block">
              <div className="list-row list-row--stacked">
                <span>
                  <strong>Active bidders:</strong>{" "}
                  {state.auction.active_player_names.join(", ")}
                </span>
                <span>
                  <strong>Passed:</strong>{" "}
                  {state.auction.passed_player_names.length > 0
                    ? state.auction.passed_player_names.join(", ")
                    : "Nobody has passed"}
                </span>
              </div>
            </div>

            {state.phase === "auction" ? (
              <div className="auction-actions">
                {bidActions.map((action) => (
                  <button
                    key={action.amount}
                    type="button"
                    onClick={() => handleBid(action.amount)}
                    disabled={isLoading}
                  >
                    Bid {action.amount}
                  </button>
                ))}
                {canPassAuction ? (
                  <button
                    type="button"
                    onClick={handlePassAuction}
                    disabled={isLoading}
                  >
                    Pass
                  </button>
                ) : (
                  <span className="muted">
                    At least one player must bid before the auction can end.
                  </span>
                )}
              </div>
            ) : (
              <p className="muted">
                Auction complete. {state.auction.highest_bidder_name} won at{" "}
                {state.auction.current_high_bid}.
              </p>
            )}

            <div className="list-block">
              <strong>Bid history</strong>
              {state.auction.bid_history.length === 0 ? (
                <span className="muted">No bids or passes yet.</span>
              ) : (
                state.auction.bid_history.map((event, index) => (
                  <div
                    key={`${event.bidder_name}-${event.action}-${index}`}
                    className="list-row"
                  >
                    <span>{event.bidder_name}</span>
                    <span>
                      {event.action === "bid"
                        ? `bid ${event.amount}`
                        : "passed"}
                    </span>
                  </div>
                ))
              )}
            </div>
          </section>

          <section className="panel">
            <h2>Players</h2>
            <div className="players-grid">
              {state.round.players.map((player) => (
                <PlayerPanel
                  key={player.name}
                  player={player}
                  trump={state.round.trump}
                  isCurrentPlayer={state.phase === "play" &&
                    player.name === state.round.current_player_name
                  }
                />
              ))}
            </div>
          </section>

          {state.phase !== "auction" ? (
            <>
              <CurrentTrickPanel trick={state.round.current_trick} />

              <LegalActionsPanel
                actions={playActions}
                currentPlayer={currentPlayer}
                trump={state.round.trump}
                isTerminal={state.round.is_terminal}
                disabled={isLoading}
                onPlay={handlePlay}
              />
            </>
          ) : null}

          <section className="panel">
            <h2>Trick History</h2>
            <div className="list-block">
              {state.round.trick_history.length === 0 ? (
                <span className="muted">No completed tricks yet.</span>
              ) : (
                state.round.trick_history.map((trick, index) => (
                  <div
                    key={`${trick.leader_name}-${index}`}
                    className="history-block"
                  >
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
              <p className="muted">
                Score is only available once the round is terminal.
              </p>
            ) : score ? (
              <>
                <div className="score-awards">
                  <div className="score-award-card">
                    <strong>High</strong>
                    <span>
                      {score.awards.high.unit_name} with {score.high_card.code}
                    </span>
                  </div>
                  <div className="score-award-card">
                    <strong>Jack</strong>
                    <span>
                      {score.awards.jack.unit_name && score.awards.jack.card
                        ? `${score.awards.jack.unit_name} with ${score.awards.jack.card.code}`
                        : (score.awards.jack.reason ?? "No jack point awarded")}
                    </span>
                  </div>
                  <div className="score-award-card">
                    <strong>Low</strong>
                    <span>
                      {score.awards.low.unit_name} via{" "}
                      {score.awards.low.player_name} with {score.low_card.code}
                    </span>
                  </div>
                  <div className="score-award-card">
                    <strong>Game</strong>
                    <span>
                      {score.awards.game.unit_name
                        ? `${score.awards.game.unit_name} (${score.awards.game.game_total})`
                        : score.awards.game.tied_unit_names &&
                            score.awards.game.tied_unit_names.length > 0
                          ? `No point awarded, tie at ${score.awards.game.game_total}: ${score.awards.game.tied_unit_names.join(", ")}`
                          : "No game point awarded"}
                    </span>
                  </div>
                </div>

                {state.match.is_complete ? (
                  <p className="match-complete-banner">
                    Match complete. Winner: {state.match.winner_names.join(", ")}.
                    Start a new game to play again.
                  </p>
                ) : null}

                <div className="score-grid">
                  {score.results.map((result) => (
                    <div
                      key={result.name}
                      className="score-cell score-cell--detailed"
                    >
                      <strong>{result.name}</strong>
                      {result.member_names.length > 1 ? (
                        <span className="muted">
                          {result.member_names.join(", ")}
                        </span>
                      ) : null}
                      <span>Total points: {result.total_points}</span>
                      <span>High: {result.breakdown.high}</span>
                      <span>Jack: {result.breakdown.jack}</span>
                      <span>Low: {result.breakdown.low}</span>
                      <span>Jokers: {result.breakdown.jokers}</span>
                      <span>Game: {result.breakdown.game}</span>
                      <span>Game total: {result.game_total}</span>
                    </div>
                  ))}
                </div>
              </>
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
                    <strong>Team {index + 1}:</strong>{" "}
                    {team.constituents.join(", ")}
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
