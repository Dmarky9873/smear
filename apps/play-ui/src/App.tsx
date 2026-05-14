import { useState } from "react";

import {
  buildDisplayCapturedByPlayer,
  loadOrCreateSessionId,
  PlayingCard,
  sortCardsHighToLow,
  useSmearGame,
  type Card,
  type TrickState,
} from "@smear/web-core";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEFAULT_TEAMS_INPUT = "You, North\nEast, West";

function getRoundBanner(phase: string, currentTurnName: string): string {
  if (phase === "auction") {
    return `Bid: ${currentTurnName}`;
  }

  if (phase === "play") {
    return `Turn: ${currentTurnName}`;
  }

  if (phase === "match_complete") {
    return "Match over";
  }

  return "Round over";
}

function getVisiblePlayForPlayer(
  trick: TrickState | null,
  playerName: string,
) {
  return trick?.plays.find((play) => play.player_name === playerName) ?? null;
}

function getScoreHeading(roundIsTerminal: boolean): string {
  return roundIsTerminal ? "Round Summary" : "Last Round Summary";
}

function getJokerAwardText(results: Array<{ name: string; joker_count: number }>) {
  const jokerWinners = results
    .filter((result) => result.joker_count > 0)
    .map((result) => `${result.name} (${result.joker_count})`);

  if (jokerWinners.length === 0) {
    return "None";
  }

  return jokerWinners.join(", ");
}

export default function App() {
  const [sessionId] = useState(() =>
    loadOrCreateSessionId("smear-play-session-id"),
  );
  const [showHistory, setShowHistory] = useState(false);
  const [teamsEnabled, setTeamsEnabled] = useState(true);
  const [savedTeamsInput, setSavedTeamsInput] = useState(DEFAULT_TEAMS_INPUT);

  const {
    availableBots,
    awaitingNextTrick,
    bidActions,
    botProgress,
    botThinkingName,
    canPassAuction,
    currentTurnName,
    error,
    handleAdvanceToNextTrick,
    handleBidPlay,
    handleNewGamePlay,
    handleNextRoundPlay,
    handlePassAuctionPlay,
    handlePlayPlay,
    handlePlayerBotChange,
    handlePlayerNameChange,
    isLoading,
    numPlayers,
    playActions,
    playerBots,
    playerNames,
    revealedTrick,
    score,
    setNumPlayers,
    setTeamsInput,
    state,
    teamsInput,
    turnPlayer,
  } = useSmearGame({
    apiBaseUrl: API_BASE_URL,
    botAutomationEnabled: true,
    initialBotActionDelayMs: 700,
    initialNumPlayers: 4,
    initialPlayerBots: [null, "greedy", "greedy", "greedy"],
    initialPlayerNames: ["You", "North", "East", "West"],
    initialTeamsInput: DEFAULT_TEAMS_INPUT,
    sessionId,
  });

  const legalCardCodes = new Set(playActions.map((action) => action.card_code));
  const isHumanTurn = Boolean(turnPlayer && !turnPlayer.bot_id);
  const visibleTrick = state
    ? state.round.current_trick.plays.length > 0
      ? state.round.current_trick
      : revealedTrick ?? state.round.current_trick
    : null;
  const completedTricks = state ? [...state.round.trick_history].reverse() : [];
  const displayCapturedByPlayer = state
    ? buildDisplayCapturedByPlayer(state.round)
    : null;
  const orderedTurnCards = turnPlayer
    ? sortCardsHighToLow(turnPlayer.cards)
    : [];
  const showBotProgress = Boolean(botThinkingName) && !awaitingNextTrick;
  const progressPercent = Math.max(
    0,
    Math.min(100, botProgress?.percent_complete ?? 0),
  );

  function confirmNewGameIfNeeded() {
    if (!state) {
      return true;
    }

    return window.confirm("Start a fresh table with the current setup?");
  }

  async function handleStartTable() {
    if (!confirmNewGameIfNeeded()) {
      return;
    }

    await handleNewGamePlay();
  }

  function handleTeamsEnabledChange(enabled: boolean) {
    setTeamsEnabled(enabled);

    if (enabled) {
      setTeamsInput(savedTeamsInput.trim() ? savedTeamsInput : DEFAULT_TEAMS_INPUT);
      return;
    }

    if (teamsInput.trim()) {
      setSavedTeamsInput(teamsInput);
    }
    setTeamsInput("");
  }

  function handleTeamsInputChange(value: string) {
    setTeamsInput(value);
    setSavedTeamsInput(value);
  }

  return (
    <main className="public-shell">
      <section className="hero-card">
        <h1>smear online alpha</h1>
      </section>

      {error ? <div className="banner banner--error">{error}</div> : null}

      <div className="experience-grid">
        <aside className="control-card">
          <div className="card-header">
            <div>
              <span className="eyebrow">Table setup</span>
              <h2>Players and teams</h2>
            </div>
          </div>

          <label className="toggle-row">
            <input
              type="checkbox"
              checked={teamsEnabled}
              onChange={(event) =>
                handleTeamsEnabledChange(event.target.checked)
              }
            />
            <span>Play with teams</span>
          </label>

          <label className="field">
            <span>Seats</span>
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

          <div className="seat-editor">
            {playerNames.map((playerName, index) => (
              <article key={index} className="seat-editor__card">
                <label className="field">
                  <span>Seat {index + 1}</span>
                  <input
                    type="text"
                    value={playerName}
                    onChange={(event) =>
                      handlePlayerNameChange(index, event.target.value)
                    }
                  />
                </label>
                <label className="field">
                  <span>Controller</span>
                  <select
                    value={playerBots[index] ?? ""}
                    onChange={(event) =>
                      handlePlayerBotChange(index, event.target.value || null)
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
              </article>
            ))}
          </div>

          {teamsEnabled ? (
            <label className="field">
              <span>Teams</span>
              <textarea
                rows={4}
                value={teamsInput}
                onChange={(event) => handleTeamsInputChange(event.target.value)}
                placeholder={DEFAULT_TEAMS_INPUT}
              />
            </label>
          ) : null}

          <button
            type="button"
            className="cta-button"
            onClick={() => {
              void handleStartTable();
            }}
            disabled={isLoading}
          >
            New game
          </button>
        </aside>

        <section className="board-column">
          {!state ? (
            <section className="empty-state-card">
              <span className="eyebrow">No active table</span>
              <h2>Start a new game from the table setup.</h2>
            </section>
          ) : (
            <>
              {showBotProgress ? (
                <section className="progress-card" aria-live="polite">
                  <div className="progress-card__header">
                    <span>Bot thinking</span>
                    <strong>{botThinkingName}</strong>
                  </div>
                  <div
                    className={[
                      "progress-track",
                      botProgress ? "" : "is-indeterminate",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    role="progressbar"
                    aria-label={`${botThinkingName} is evaluating moves`}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={botProgress ? progressPercent : undefined}
                  >
                    <div
                      className="progress-bar"
                      style={botProgress ? { width: `${progressPercent}%` } : undefined}
                    />
                  </div>
                  <p className="help-copy">
                    {botProgress
                      ? `${botProgress.label ?? "Search in progress"}: ${botProgress.completed_units ?? 0}/${botProgress.total_units ?? 0}${
                          botProgress.detail ? `, ${botProgress.detail}` : ""
                        }`
                      : "Smear is evaluating the next move."}
                  </p>
                </section>
              ) : null}

              <section className="status-row">
                <div className="status-pill">
                  <span>Round</span>
                  <strong>{state.match.round_number}</strong>
                </div>
                <div className="status-pill">
                  <span>Trump</span>
                  <strong>{state.round.trump ?? "-"}</strong>
                </div>
                <div className="status-pill status-pill--wide">
                  <span>Status</span>
                  <strong>
                    {awaitingNextTrick
                      ? "Next trick ready"
                      : botThinkingName ?? getRoundBanner(state.phase, currentTurnName)}
                  </strong>
                </div>
                {state.match.scores.map((entry) => (
                  <div key={entry.name} className="status-pill">
                    <span>{entry.name}</span>
                    <strong>{entry.points}</strong>
                  </div>
                ))}
              </section>

              <section className="felt-card">
                <div className="card-header">
                  <div>
                    <span className="eyebrow">Table</span>
                    <h2>{state.phase === "auction" ? "Auction" : "Current trick"}</h2>
                  </div>
                  <div className="card-header__actions">
                    {awaitingNextTrick ? (
                      <button
                        type="button"
                        className="ghost-button"
                        onClick={handleAdvanceToNextTrick}
                        disabled={isLoading}
                      >
                        Continue
                      </button>
                    ) : state.round.is_terminal && !state.match.is_complete ? (
                      <button
                        type="button"
                        className="ghost-button"
                        onClick={() => {
                          void handleNextRoundPlay();
                        }}
                        disabled={isLoading}
                      >
                        Next round
                      </button>
                    ) : null}
                  </div>
                </div>

                <div className="seat-grid">
                  {state.round.players.map((player) => {
                    const isCurrentPlayer = player.name === turnPlayer?.name;
                    const tablePlay = getVisiblePlayForPlayer(visibleTrick, player.name);
                    const displayCaptured = displayCapturedByPlayer?.[player.name] ?? {
                      cards: player.captured_cards,
                      count: player.captured_count,
                    };

                    return (
                      <article
                        key={player.name}
                        className={[
                          "seat-card",
                          isCurrentPlayer ? "seat-card--current" : "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                      >
                        <div className="seat-card__header">
                          <div>
                            <h3>{player.name}</h3>
                            <p>
                              {player.bot_label ? player.bot_label : "Human"}
                            </p>
                          </div>
                          <span className="seat-card__meta">
                            {player.cards.length} cards
                          </span>
                        </div>

                        <div className="seat-table">
                          {tablePlay ? (
                            <PlayingCard
                              card={tablePlay.card}
                              compact
                              isTrump={
                                !tablePlay.card.is_joker &&
                                state.round.trump === tablePlay.card.suit
                              }
                            />
                          ) : (
                            <div className="seat-table__empty" />
                          )}
                        </div>

                        <div className="seat-card__footer">
                          <span>Captured {displayCaptured.count}</span>
                          <span>
                            {state.auction.highest_bidder_name === player.name &&
                            state.auction.current_high_bid !== null
                              ? `Bid ${state.auction.current_high_bid}`
                              : " "}
                          </span>
                        </div>

                        <div className="seat-captured">
                          <span className="seat-captured__label">
                            Captured cards
                          </span>
                          <div
                            className="seat-captured__cards"
                            aria-label={`${player.name} captured cards`}
                          >
                            {displayCaptured.cards.length > 0 ? (
                              displayCaptured.cards.map((card: Card, index) => (
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
                              <span className="seat-captured__empty">None</span>
                            )}
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>

                <div className="table-actions">
                  {state.phase === "auction" ? (
                    <>
                      <div className="table-copy">
                        <span>Current bid</span>
                        <strong>{state.auction.current_high_bid ?? "-"}</strong>
                      </div>
                      {isHumanTurn ? (
                        <div className="button-row">
                          {bidActions.map((action) => (
                            <button
                              key={action.amount}
                              type="button"
                              onClick={() => {
                                void handleBidPlay(action.amount);
                              }}
                              disabled={isLoading}
                            >
                              Bid {action.amount}
                            </button>
                          ))}
                          {canPassAuction ? (
                            <button
                              type="button"
                              onClick={() => {
                                void handlePassAuctionPlay();
                              }}
                              disabled={isLoading}
                            >
                              Pass
                            </button>
                          ) : null}
                        </div>
                      ) : (
                        <p className="help-copy">Waiting for {currentTurnName} to act.</p>
                      )}
                    </>
                  ) : (
                    <div className="table-copy">
                      <span>Lead</span>
                      <strong>{visibleTrick?.leader_name ?? "-"}</strong>
                      <span>Winner</span>
                      <strong>{visibleTrick?.winner_name ?? "-"}</strong>
                    </div>
                  )}
                </div>
              </section>

              <section className="hand-card">
                <div className="card-header">
                  <div>
                    <span className="eyebrow">Your hand</span>
                    <h2>
                      {turnPlayer && !turnPlayer.bot_id
                        ? turnPlayer.name
                        : "Waiting on bots"}
                    </h2>
                  </div>
                </div>

                {turnPlayer && !turnPlayer.bot_id ? (
                  <div className="hand-grid">
                    {orderedTurnCards.map((card: Card) => (
                      <PlayingCard
                        key={card.code}
                        card={card}
                        isTrump={!card.is_joker && state.round.trump === card.suit}
                        disabled={
                          state.phase === "play"
                            ? awaitingNextTrick ||
                              isLoading ||
                              !legalCardCodes.has(card.code)
                            : true
                        }
                        onClick={
                          state.phase === "play" && legalCardCodes.has(card.code)
                            ? () => {
                                void handlePlayPlay(card.code);
                              }
                            : undefined
                        }
                      />
                    ))}
                  </div>
                ) : (
                  <div className="empty-inline">
                    {botThinkingName
                      ? `${botThinkingName} is playing`
                      : "No human hand is active right now."}
                  </div>
                )}
              </section>

              {completedTricks.length > 0 ? (
                <section className="history-card">
                  <div className="card-header">
                    <div>
                      <span className="eyebrow">Trick history</span>
                      <h2>{completedTricks.length} completed</h2>
                    </div>
                    <button
                      type="button"
                      className="ghost-button"
                      onClick={() => setShowHistory((current) => !current)}
                    >
                      {showHistory ? "Hide" : "Show"}
                    </button>
                  </div>

                  {showHistory ? (
                    <div className="history-list">
                      {completedTricks.map((trick, trickIndex) => (
                        <article
                          key={`${trick.leader_name}-${trickIndex}-${trick.winner_name ?? "unknown"}`}
                          className="history-item"
                        >
                          <div className="history-item__header">
                            <span>Lead {trick.leader_name}</span>
                            <strong>{trick.winner_name ?? "-"}</strong>
                          </div>
                          <div className="history-item__cards">
                            {trick.plays.map((play, index) => (
                              <div
                                key={`${play.player_name}-${play.card.code}-${index}`}
                                className="history-play"
                              >
                                <span>{play.player_name}</span>
                                <PlayingCard card={play.card} compact />
                              </div>
                            ))}
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : null}
                </section>
              ) : null}

              {score ? (
                <section className="summary-card">
                  <div className="card-header">
                    <div>
                      <span className="eyebrow">Scoring</span>
                      <h2>{getScoreHeading(state.round.is_terminal)}</h2>
                    </div>
                    {state.match.is_complete ? (
                      <div className="hero-badge">
                        <span>Winner</span>
                        <strong>{state.match.winner_names.join(", ")}</strong>
                      </div>
                    ) : null}
                  </div>

                  <div className="award-grid">
                    <div className="award-card">
                      <span>High</span>
                      <strong>{score.awards.high.unit_name}</strong>
                    </div>
                    <div className="award-card">
                      <span>Jack</span>
                      <strong>
                        {score.awards.jack.unit_name ?? score.awards.jack.reason}
                      </strong>
                    </div>
                    <div className="award-card">
                      <span>Low</span>
                      <strong>{score.awards.low.unit_name}</strong>
                    </div>
                    <div className="award-card">
                      <span>Jokers</span>
                      <strong>{getJokerAwardText(score.results)}</strong>
                    </div>
                    <div className="award-card">
                      <span>Game</span>
                      <strong>
                        {score.awards.game.unit_name ??
                          score.awards.game.tied_unit_names?.join(", ") ??
                          "-"}
                      </strong>
                    </div>
                  </div>

                  <div className="result-grid">
                    {score.results.map((result) => (
                      <article key={result.name} className="result-card">
                        <div className="result-card__header">
                          <h3>{result.name}</h3>
                          <strong>{result.match_delta > 0 ? `+${result.match_delta}` : result.match_delta}</strong>
                        </div>
                        <p>
                          High {result.breakdown.high} | Jack {result.breakdown.jack} |
                          Low {result.breakdown.low} | Jokers {result.breakdown.jokers} |
                          Game {result.breakdown.game}
                        </p>
                      </article>
                    ))}
                  </div>
                </section>
              ) : null}
            </>
          )}
        </section>
      </div>
    </main>
  );
}
