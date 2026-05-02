import { useEffect, useState } from "react";

import { PlayingCard } from "./PlayingCard";
import type {
  BotProgress,
  BidAction,
  Card,
  GameState,
  Play,
  PlayCardAction,
  Player,
  ReadyBot,
  Score,
  TrickState,
} from "../types";

type PlayModeViewProps = {
  numPlayers: number;
  playerNames: string[];
  playerBots: (string | null)[];
  teamsInput: string;
  availableBots: ReadyBot[];
  botProgress: BotProgress | null;
  state: GameState | null;
  score: Score | null;
  error: string | null;
  isBusy: boolean;
  botThinkingName: string | null;
  revealedTrick: TrickState | null;
  awaitingNextTrick: boolean;
  botActionDelayMs: number;
  currentTurnName: string;
  turnPlayer: Player | null;
  playActions: PlayCardAction[];
  bidActions: BidAction[];
  canPassAuction: boolean;
  onNumPlayersChange: (value: number) => void;
  onPlayerNameChange: (index: number, value: string) => void;
  onPlayerBotChange: (index: number, value: string | null) => void;
  onTeamsInputChange: (value: string) => void;
  onBotActionDelayChange: (value: number) => void;
  onAdvanceToNextTrick: () => void;
  onNewGame: () => void;
  onNextRound: () => void;
  onPlay: (cardCode: string) => Promise<void>;
  onBid: (amount: number) => void;
  onPassAuction: () => void;
};

function getRoundBanner(state: GameState, currentTurnName: string): string {
  if (state.phase === "auction") {
    return `Bid: ${currentTurnName}`;
  }

  if (state.phase === "play") {
    return `Turn: ${currentTurnName}`;
  }

  if (state.phase === "match_complete") {
    return "Match over";
  }

  return "Round over";
}

function getVisiblePlayForPlayer(
  trick: TrickState | null,
  playerName: string,
): Play | null {
  return trick?.plays.find((play) => play.player_name === playerName) ?? null;
}

function getCompletedTricks(state: GameState): TrickState[] {
  return [...state.round.trick_history].reverse();
}

export function PlayModeView({
  numPlayers,
  playerNames,
  playerBots,
  teamsInput,
  availableBots,
  botProgress,
  state,
  score,
  error,
  isBusy,
  botThinkingName,
  revealedTrick,
  awaitingNextTrick,
  botActionDelayMs,
  currentTurnName,
  turnPlayer,
  playActions,
  bidActions,
  canPassAuction,
  onNumPlayersChange,
  onPlayerNameChange,
  onPlayerBotChange,
  onTeamsInputChange,
  onBotActionDelayChange,
  onAdvanceToNextTrick,
  onNewGame,
  onNextRound,
  onPlay,
  onBid,
  onPassAuction,
}: PlayModeViewProps) {
  const [showTeamsEditor, setShowTeamsEditor] = useState(Boolean(teamsInput));
  const [showHistory, setShowHistory] = useState(false);
  const legalCardCodes = new Set(playActions.map((action) => action.card_code));
  const isHumanTurn = Boolean(turnPlayer && !turnPlayer.bot_id);
  const completedTricks = state ? getCompletedTricks(state) : [];
  const visibleTrick = state
    ? state.round.current_trick.plays.length > 0
      ? state.round.current_trick
      : revealedTrick ?? state.round.current_trick
    : null;
  const canShowCurrentHand =
    Boolean(state) &&
    Boolean(turnPlayer) &&
    !turnPlayer?.bot_id &&
    (state?.phase === "auction" || state?.phase === "play");
  const showBotProgress = Boolean(botThinkingName) && !awaitingNextTrick;
  const progressPercent = Math.max(
    0,
    Math.min(100, botProgress?.percent_complete ?? 0),
  );

  useEffect(() => {
    setShowHistory(false);
  }, [state?.match.round_number]);

  return (
    <main className="play-shell">
      {error ? <div className="error-banner">{error}</div> : null}

      <section className="play-setup-card">
        <div className="play-section-heading">
          <h2>Play</h2>
          <div className="play-section-heading__side">
            <label className="play-delay-control">
              <span>Bot pause</span>
              <div className="play-delay-control__row">
                <input
                  className="play-delay-control__slider"
                  type="range"
                  min={0}
                  max={2000}
                  step={100}
                  value={botActionDelayMs}
                  onChange={(event) =>
                    onBotActionDelayChange(Number(event.target.value))
                  }
                />
                <strong>{botActionDelayMs} ms</strong>
              </div>
            </label>
            <button type="button" onClick={onNewGame} disabled={isBusy}>
              {state ? "New game" : "Start"}
            </button>
          </div>
        </div>

        <div className="play-setup-grid">
          <label className="play-field">
            <span>Players</span>
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
            <div key={index} className="play-player-card">
              <label className="play-field">
                <span>Name</span>
                <input
                  type="text"
                  value={playerName}
                  onChange={(event) =>
                    onPlayerNameChange(index, event.target.value)
                  }
                />
              </label>
              <label className="play-field">
                <span>Type</span>
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
            </div>
          ))}
        </div>

        <div className="play-setup-footer">
          <button
            type="button"
            className="play-link-button"
            onClick={() => setShowTeamsEditor((current) => !current)}
          >
            Teams
          </button>
        </div>

        {showTeamsEditor ? (
          <label className="play-field play-field--full">
            <span>Teams</span>
            <textarea
              rows={4}
              value={teamsInput}
              onChange={(event) => onTeamsInputChange(event.target.value)}
              placeholder={"Player 1, Player 3\nPlayer 2, Player 4"}
            />
          </label>
        ) : null}
      </section>

      {!state ? null : (
        <>
          {showBotProgress ? (
            <section className="play-progress-card" aria-live="polite">
              <div className="play-progress-card__header">
                <span>Bot thinking</span>
                <strong>{botThinkingName}</strong>
              </div>
              <div
                className={[
                  "play-progress-track",
                  botProgress ? "" : "is-indeterminate",
                ]
                  .filter(Boolean)
                  .join(" ")}
                role="progressbar"
                aria-label={`${botThinkingName} is evaluating moves`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={botProgress ? progressPercent : undefined}
                aria-valuetext={
                  botProgress
                    ? `${Math.round(progressPercent)} percent complete`
                    : "Search in progress"
                }
              >
                <div
                  className="play-progress-bar"
                  style={botProgress ? { width: `${progressPercent}%` } : undefined}
                />
              </div>
              <p className="play-progress-copy">
                {botProgress
                  ? `${botProgress.label ?? "Search in progress"}: ${botProgress.completed_units ?? 0}/${botProgress.total_units ?? 0}${
                      botProgress.detail ? `, ${botProgress.detail}` : ""
                    }`
                  : "Higher-depth minimax searches can take a while."}
              </p>
            </section>
          ) : null}

          <section className="play-status-row">
            <div className="play-score-pill">
              <span>Round</span>
              <strong>{state.match.round_number}</strong>
            </div>
            <div className="play-score-pill">
              <span>Trump</span>
              <strong>{state.round.trump ?? "-"}</strong>
            </div>
            <div className="play-score-pill">
              <span>Status</span>
              <strong>
                {awaitingNextTrick
                  ? "Next trick"
                  : botThinkingName ?? getRoundBanner(state, currentTurnName)}
              </strong>
            </div>
            {state.match.scores.map((entry) => (
              <div key={entry.name} className="play-score-pill">
                <span>{entry.name}</span>
                <strong>{entry.points}</strong>
              </div>
            ))}
          </section>

          <section className="play-hand-card play-hand-card--sticky">
            <div className="play-section-heading">
              <h2>Hand</h2>
              {awaitingNextTrick ? (
                <button
                  type="button"
                  onClick={onAdvanceToNextTrick}
                  disabled={isBusy}
                >
                  Move to next trick
                </button>
              ) : state.round.is_terminal && !state.match.is_complete ? (
                <button type="button" onClick={onNextRound} disabled={isBusy}>
                  Next round
                </button>
              ) : null}
            </div>

            {canShowCurrentHand && turnPlayer ? (
              <div className="play-hand-grid">
                {turnPlayer.cards.map((card) => (
                  <PlayingCard
                    key={card.code}
                    card={card}
                    isTrump={!card.is_joker && state.round.trump === card.suit}
                    disabled={
                      state.phase === "play"
                        ? awaitingNextTrick ||
                          isBusy ||
                          !legalCardCodes.has(card.code)
                        : true
                    }
                    onClick={
                      state.phase === "play" && legalCardCodes.has(card.code)
                        ? () => onPlay(card.code)
                        : undefined
                    }
                    className="play-hand-card__item"
                  />
                ))}
              </div>
            ) : (
              <div className="play-hand-empty">
                {botThinkingName ? "Bot is playing" : "Waiting"}
              </div>
            )}
          </section>

          <section className="play-table-card">
            <div className="play-seat-grid">
              {state.round.players.map((player) => {
                const isCurrentPlayer = player.name === turnPlayer?.name;
                const isBot = Boolean(player.bot_id);
                const tablePlay = getVisiblePlayForPlayer(visibleTrick, player.name);
                const isAuctionWinner =
                  state.auction.highest_bidder_name === player.name &&
                  state.auction.current_high_bid !== null;
                const showTableCardSection = state.phase !== "auction";
                const showCapturedCards =
                  state.phase !== "auction" || player.captured_cards.length > 0;

                return (
                  <article
                    key={player.name}
                    className={[
                      "play-seat",
                      isCurrentPlayer ? "play-seat--current" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    {isAuctionWinner ? (
                      <div className="play-seat__bid-badge">
                        Bid {state.auction.current_high_bid}
                      </div>
                    ) : null}
                    <div className="play-seat__header">
                      <h3>
                        {player.name}
                        {player.bot_label ? ` (${player.bot_label})` : ""}
                      </h3>
                      <span className="play-seat__status">
                        {isBot ? "Bot" : "Human"}
                      </span>
                    </div>
                    <div className="play-seat__meta">
                      <span>{player.cards.length} cards</span>
                      <span>{player.captured_count} won</span>
                    </div>

                    {showTableCardSection ? (
                      <div className="play-seat__section">
                        <span className="play-seat__label">Table</span>
                        <div className="play-seat__table-slot">
                          {tablePlay ? (
                            <PlayingCard
                              card={tablePlay.card}
                              compact
                              isTrump={
                                !tablePlay.card.is_joker &&
                                state.round.trump === tablePlay.card.suit
                              }
                              className="play-seat__table-card"
                            />
                          ) : (
                            <div
                              className="play-seat__table-empty"
                              aria-label={`${player.name} has not played this trick`}
                            />
                          )}
                        </div>
                      </div>
                    ) : null}

                    {showCapturedCards ? (
                      <>
                        <div className="play-seat__section">
                          <span className="play-seat__label">Won</span>
                          <div className="play-seat__cards">
                            {player.captured_cards.length > 0 ? (
                              player.captured_cards.map((card: Card, index) => (
                                <PlayingCard
                                  key={`${player.name}-${card.code}-${index}`}
                                  card={card}
                                  compact
                                  isTrump={
                                    !card.is_joker &&
                                    state.round.trump === card.suit
                                  }
                                />
                              ))
                            ) : (
                              <span className="muted">-</span>
                            )}
                          </div>
                        </div>
                      </>
                    ) : null}
                  </article>
                );
              })}
            </div>

            <div className="play-board-grid">
              <section className="play-centerboard">
                {state.phase === "auction" ? (
                  <>
                    <div className="play-centerboard__header">
                      <h3>Bid</h3>
                      <strong>{state.auction.current_high_bid ?? "-"}</strong>
                    </div>

                    {isHumanTurn ? (
                      <div className="play-action-row">
                        {bidActions.map((action) => (
                          <button
                            key={action.amount}
                            type="button"
                            onClick={() => onBid(action.amount)}
                            disabled={isBusy}
                          >
                            {action.amount}
                          </button>
                        ))}
                        {canPassAuction ? (
                          <button
                            type="button"
                            onClick={onPassAuction}
                            disabled={isBusy}
                          >
                            Pass
                          </button>
                        ) : null}
                      </div>
                    ) : null}

                    <div className="play-history-list">
                      {state.auction.bid_history.map((event, index) => (
                        <div
                          key={`${event.bidder_name}-${event.action}-${index}`}
                          className="play-history-row"
                        >
                          <span>{event.bidder_name}</span>
                          <strong>
                            {event.action === "bid" ? event.amount : "Pass"}
                          </strong>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="play-centerboard__header">
                      <h3>Table</h3>
                      <strong>
                        {visibleTrick?.winner_name
                          ? `Won by ${visibleTrick.winner_name}`
                          : visibleTrick?.leader_name
                            ? `Lead ${visibleTrick.leader_name}`
                            : "-"}
                      </strong>
                    </div>

                    <div className="play-table-summary">
                      <div className="play-table-summary__item">
                        <span>Lead</span>
                        <strong>{visibleTrick?.leader_name ?? "-"}</strong>
                      </div>
                      <div className="play-table-summary__item">
                        <span>Winner</span>
                        <strong>{visibleTrick?.winner_name ?? "-"}</strong>
                      </div>
                    </div>

                    <div className="play-table-empty-wrap">
                      {visibleTrick && visibleTrick.plays.length > 0 ? (
                        <div className="play-table-note">
                          Cards stay under each player.
                        </div>
                      ) : (
                        <div className="play-hand-empty">Waiting for the next card</div>
                      )}
                    </div>

                    {completedTricks.length > 0 ? (
                      <div className="play-trick-history">
                        <div className="play-history-toggle">
                          <button
                            type="button"
                            onClick={() => setShowHistory((current) => !current)}
                          >
                            {showHistory
                              ? "Hide history"
                              : `Show history (${completedTricks.length})`}
                          </button>
                        </div>
                        {showHistory ? (
                          <div className="play-trick-history__list">
                            {completedTricks.map((trick, trickIndex) => (
                              <div
                                key={`${trick.leader_name}-${trickIndex}-${trick.winner_name ?? "unknown"}`}
                                className="play-history-block"
                              >
                                <div className="play-history-block__header">
                                  <span>Winner</span>
                                  <strong>{trick.winner_name ?? "-"}</strong>
                                </div>
                                <div className="play-history-block__cards">
                                  {trick.plays.map((play, index) => (
                                    <div
                                      key={`${play.player_name}-${play.card.code}-${index}`}
                                      className="play-trick-card"
                                    >
                                      <span>{play.player_name}</span>
                                      <PlayingCard
                                        card={play.card}
                                        compact
                                        className="play-trick-card__face"
                                      />
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </>
                )}
              </section>
            </div>
          </section>

          {state.round.is_terminal && score ? (
            <section className="play-summary-card">
              <div className="play-section-heading">
                <h2>Score</h2>
                {state.match.is_complete ? (
                  <strong>{state.match.winner_names.join(", ")}</strong>
                ) : null}
              </div>

              <div className="play-awards-grid">
                <div className="play-mini-card">
                  <span>High</span>
                  <strong>{score.awards.high.unit_name}</strong>
                </div>
                <div className="play-mini-card">
                  <span>Jack</span>
                  <strong>
                    {score.awards.jack.unit_name ?? score.awards.jack.reason}
                  </strong>
                </div>
                <div className="play-mini-card">
                  <span>Low</span>
                  <strong>{score.awards.low.unit_name}</strong>
                </div>
                <div className="play-mini-card">
                  <span>Game</span>
                  <strong>
                    {score.awards.game.unit_name ??
                      score.awards.game.tied_unit_names?.join(", ") ??
                      "-"}
                  </strong>
                </div>
              </div>

              <div className="play-summary-section">
                <div className="play-summary-section__header">
                  <h3>Sleeping</h3>
                  <strong>{state.round.hidden_cards.length}</strong>
                </div>
                {state.round.hidden_cards.length > 0 ? (
                  <div className="play-summary-cards">
                    {state.round.hidden_cards.map((card, index) => (
                      <PlayingCard
                        key={`${card.code}-${index}`}
                        card={card}
                        compact
                        isTrump={
                          !card.is_joker && state.round.trump === card.suit
                        }
                      />
                    ))}
                  </div>
                ) : (
                  <div className="play-hand-empty">No sleeping cards</div>
                )}
              </div>
            </section>
          ) : null}
        </>
      )}
    </main>
  );
}
