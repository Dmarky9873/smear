import { CurrentTrickPanel } from "./CurrentTrickPanel";
import { DebugJsonPanel } from "./DebugJsonPanel";
import { LegalActionsPanel } from "./LegalActionsPanel";
import { PlayerPanel } from "./PlayerPanel";
import { PlayingCard } from "./PlayingCard";
import type {
  BidAction,
  GameState,
  PlayCardAction,
  Player,
  ReadyBot,
  Score,
} from "../types";

type DebugModeViewProps = {
  numPlayers: number;
  playerNames: string[];
  playerBots: (string | null)[];
  teamsInput: string;
  availableBots: ReadyBot[];
  state: GameState | null;
  score: Score | null;
  error: string | null;
  isLoading: boolean;
  currentTurnName: string;
  currentPlayer: Player | null;
  playActions: PlayCardAction[];
  bidActions: BidAction[];
  canPassAuction: boolean;
  onNumPlayersChange: (value: number) => void;
  onPlayerNameChange: (index: number, value: string) => void;
  onPlayerBotChange: (index: number, value: string | null) => void;
  onTeamsInputChange: (value: string) => void;
  onNewGame: () => void;
  onResetRound: () => void;
  onNextRound: () => void;
  onRefreshState: () => void;
  onPlay: (cardCode: string) => Promise<void>;
  onBid: (amount: number) => void;
  onPassAuction: () => void;
};

export function DebugModeView({
  numPlayers,
  playerNames,
  playerBots,
  teamsInput,
  availableBots,
  state,
  score,
  error,
  isLoading,
  currentTurnName,
  currentPlayer,
  playActions,
  bidActions,
  canPassAuction,
  onNumPlayersChange,
  onPlayerNameChange,
  onPlayerBotChange,
  onTeamsInputChange,
  onNewGame,
  onResetRound,
  onNextRound,
  onRefreshState,
  onPlay,
  onBid,
  onPassAuction,
}: DebugModeViewProps) {
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
                onNumPlayersChange(Number(event.target.value) || 3)
              }
            />
          </label>

          {playerNames.map((playerName, index) => (
            <div key={index} className="setup-player-card">
              <label>
                Player {index + 1}
                <input
                  type="text"
                  value={playerName}
                  onChange={(event) =>
                    onPlayerNameChange(index, event.target.value)
                  }
                />
              </label>
              <label>
                Controller
                <select
                  value={playerBots[index] ?? ""}
                  onChange={(event) =>
                    onPlayerBotChange(index, event.target.value || null)
                  }
                >
                  <option value="">Human</option>
                  {availableBots.map((bot) => (
                    <option key={bot.id} value={bot.id}>
                      {bot.label}
                    </option>
                  ))}
                </select>
              </label>
              {playerBots[index] ? (
                <span className="muted">
                  {
                    availableBots.find((bot) => bot.id === playerBots[index])
                      ?.description
                  }
                </span>
              ) : (
                <span className="muted">Human-controlled player.</span>
              )}
            </div>
          ))}

          <label className="setup-grid__full">
            Teams (optional, one team per line, comma separated)
            <textarea
              rows={4}
              value={teamsInput}
              onChange={(event) => onTeamsInputChange(event.target.value)}
              placeholder={"Player 1, Player 3\nPlayer 2, Player 4"}
            />
          </label>
        </div>

        <p className="muted">
          Bot turns run automatically until the next human turn or terminal
          state.
        </p>

        <div className="button-row">
          <button type="button" onClick={onNewGame} disabled={isLoading}>
            New Game
          </button>
          <button
            type="button"
            onClick={onResetRound}
            disabled={isLoading || !state}
          >
            Reset Round (Debug)
          </button>
          <button
            type="button"
            onClick={onNextRound}
            disabled={
              isLoading ||
              !state ||
              (state.phase !== "round_complete" &&
                state.phase !== "match_complete") ||
              state.match.is_complete
            }
          >
            Next Round
          </button>
          <button
            type="button"
            onClick={onRefreshState}
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
                    onClick={() => onBid(action.amount)}
                    disabled={isLoading}
                  >
                    Bid {action.amount}
                  </button>
                ))}
                {canPassAuction ? (
                  <button
                    type="button"
                    onClick={onPassAuction}
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
                  isCurrentPlayer={
                    state.phase === "play" &&
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
                onPlay={onPlay}
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
                    <strong>Bid</strong>
                    <span>
                      {score.bid_summary.bidder_name &&
                      score.bid_summary.amount !== null &&
                      score.bid_summary.points_won !== null &&
                      score.bid_summary.match_delta !== null
                        ? score.bid_summary.made_bid
                          ? `${score.bid_summary.unit_name} made ${score.bid_summary.amount} with ${score.bid_summary.points_won} and gains ${score.bid_summary.match_delta}`
                          : `${score.bid_summary.unit_name} missed ${score.bid_summary.amount} with ${score.bid_summary.points_won} and loses ${Math.abs(score.bid_summary.match_delta)}`
                        : "No bid summary available"}
                    </span>
                  </div>
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
                        : score.awards.jack.reason ??
                          "No jack point awarded"}
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
                    Match complete. Winner: {state.match.winner_names.join(", ")}
                    . Start a new game to play again.
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
                      <span>Round points: {result.total_points}</span>
                      <span>Match delta: {result.match_delta}</span>
                      {result.bid_amount !== null ? (
                        <span>
                          Bid outcome:{" "}
                          {result.made_bid
                            ? `made ${result.bid_amount}`
                            : `missed ${result.bid_amount}`}
                        </span>
                      ) : null}
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
