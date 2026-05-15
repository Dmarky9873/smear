import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";

import {
  buildDisplayCapturedByPlayer,
  createApiClient,
  loadOrCreateSessionId,
  PlayingCard,
  sortCardsHighToLow,
  useSmearGame,
  type Card,
  type LearnAction,
  type LearnChallenge,
  type TrickState,
} from "@smear/web-core";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEFAULT_TEAMS_INPUT = "You, North\nEast, West";
const DEFAULT_TEAM_SELECTIONS = [
  [0, 1],
  [2, 3],
];
const LANDING_CARDS: Card[] = [
  { code: "AH", rank: "A", suit: "H", is_joker: false },
  { code: "10C", rank: "10", suit: "C", is_joker: false },
  { code: "JD", rank: "J", suit: "D", is_joker: false },
  { code: "J1", rank: null, suit: null, is_joker: true },
];
const DONATION_PRESETS = [
  { amountCents: 300 },
  { amountCents: 500 },
  { amountCents: 1000 },
];
const DONATION_MIN_CENTS = 100;
const DONATION_MAX_CENTS = 10000;
const DONATION_CURRENCY =
  (import.meta.env.VITE_DONATION_CURRENCY ?? "CAD").toUpperCase();
const MOBILE_TABLE_QUERY =
  "(max-width: 760px), (pointer: coarse) and (max-width: 920px)";

type AppView = "home" | "play" | "learn" | "donate";

function viewFromHash(hash: string): AppView {
  const route = hash.split("?")[0];

  if (route === "#play") {
    return "play";
  }
  if (route === "#learn") {
    return "learn";
  }
  if (route === "#donate") {
    return "donate";
  }
  return "home";
}

function setHashView(view: AppView) {
  const nextHash = view === "home" ? "" : `#${view}`;
  if (window.location.hash === nextHash) {
    return;
  }
  window.location.hash = nextHash;
}

function getMobileTableMode(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  return window.matchMedia(MOBILE_TABLE_QUERY).matches;
}

function useMobileTableMode(): boolean {
  const [isMobileTable, setIsMobileTable] = useState(getMobileTableMode);

  useEffect(() => {
    const mediaQuery = window.matchMedia(MOBILE_TABLE_QUERY);
    const updateMobileTableMode = () => {
      setIsMobileTable(mediaQuery.matches);
    };

    updateMobileTableMode();
    mediaQuery.addEventListener("change", updateMobileTableMode);
    window.addEventListener("resize", updateMobileTableMode);

    return () => {
      mediaQuery.removeEventListener("change", updateMobileTableMode);
      window.removeEventListener("resize", updateMobileTableMode);
    };
  }, []);

  return isMobileTable;
}

function getActionKey(action: LearnAction): string {
  if (action.type === "bid") {
    return `bid:${action.amount}`;
  }
  if (action.type === "play_card") {
    return `play_card:${action.card_code}`;
  }
  return "pass";
}

function findCardByCode(cards: Card[], cardCode: string | undefined) {
  if (!cardCode) {
    return null;
  }
  return cards.find((card) => card.code === cardCode) ?? null;
}

function formatDonationAmount(amountCents: number): string {
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: DONATION_CURRENCY,
    currencyDisplay: "code",
  }).format(amountCents / 100);
}

function parseDonationAmount(value: string): number | null {
  const normalized = value.trim();
  if (!/^\d+(\.\d{0,2})?$/.test(normalized)) {
    return null;
  }

  const amountCents = Math.round(Number(normalized) * 100);
  if (
    amountCents < DONATION_MIN_CENTS ||
    amountCents > DONATION_MAX_CENTS
  ) {
    return null;
  }

  return amountCents;
}

function getDonationStatus(): "success" | "cancelled" | null {
  if (typeof window === "undefined") {
    return null;
  }

  const status = new URLSearchParams(window.location.search).get("donation");
  return status === "success" || status === "cancelled" ? status : null;
}

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

function getPlayerDisplayName(playerNames: string[], index: number): string {
  return playerNames[index]?.trim() || `Player ${index + 1}`;
}

function buildTeamsInputFromSelections(
  teamSelections: number[][],
  playerNames: string[],
  numPlayers: number,
): string {
  return teamSelections
    .map((team) =>
      team
        .filter((playerIndex) => playerIndex >= 0 && playerIndex < numPlayers)
        .map((playerIndex) => getPlayerDisplayName(playerNames, playerIndex))
        .join(", "),
    )
    .filter(Boolean)
    .join("\n");
}

type PageNavigationProps = {
  onNavigateHome: () => void;
  onNavigatePlay: () => void;
  onNavigateLearn: () => void;
  onNavigateDonate: () => void;
};

function PlayBotsPage({
  onNavigateHome,
  onNavigateLearn,
  onNavigateDonate,
}: PageNavigationProps) {
  const [sessionId] = useState(() =>
    loadOrCreateSessionId("smear-play-session-id"),
  );
  const [showHistory, setShowHistory] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [teamsEnabled, setTeamsEnabled] = useState(true);
  const [teamSelections, setTeamSelections] = useState<number[][]>(
    DEFAULT_TEAM_SELECTIONS,
  );
  const [setupSidebarOpen, setSetupSidebarOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const isMobileTable = useMobileTableMode();
  const gameAreaRef = useRef<HTMLElement | null>(null);

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
  const playerOptions = useMemo(
    () =>
      Array.from({ length: numPlayers }, (_, index) => ({
        index,
        name: getPlayerDisplayName(playerNames, index),
      })),
    [numPlayers, playerNames],
  );
  const assignedPlayerIndexes = useMemo(
    () => new Set(teamSelections.flat()),
    [teamSelections],
  );
  const unassignedPlayers = playerOptions.filter(
    (player) => !assignedPlayerIndexes.has(player.index),
  );
  const teamsAreComplete = !teamsEnabled || unassignedPlayers.length === 0;
  const canUseFullscreenControl = typeof document !== "undefined";
  const useMobileGameLayout = Boolean(state && isMobileTable);

  useEffect(() => {
    setTeamSelections((current) => {
      const next = current.map((team) =>
        team.filter((playerIndex) => playerIndex < numPlayers),
      );

      return next.length > 0 ? next : [[]];
    });
  }, [numPlayers]);

  useEffect(() => {
    setTeamsInput(
      teamsEnabled
        ? buildTeamsInputFromSelections(teamSelections, playerNames, numPlayers)
        : "",
    );
  }, [numPlayers, playerNames, setTeamsInput, teamSelections, teamsEnabled]);

  useEffect(() => {
    function handleFullscreenChange() {
      setIsFullscreen(document.fullscreenElement === gameAreaRef.current);
    }

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, []);

  useEffect(() => {
    function handleFullscreenEscape(event: KeyboardEvent) {
      if (
        event.key === "Escape" &&
        isFullscreen &&
        document.fullscreenElement !== gameAreaRef.current
      ) {
        setIsFullscreen(false);
      }
    }

    document.addEventListener("keydown", handleFullscreenEscape);
    return () => {
      document.removeEventListener("keydown", handleFullscreenEscape);
    };
  }, [isFullscreen]);

  useEffect(() => {
    document.body.classList.toggle("mobile-game-locked", useMobileGameLayout);

    return () => {
      document.body.classList.remove("mobile-game-locked");
    };
  }, [useMobileGameLayout]);

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
  }

  function handleTeamMemberToggle(teamIndex: number, playerIndex: number) {
    setTeamSelections((current) => {
      const selectedInTeam = current[teamIndex]?.includes(playerIndex) ?? false;
      const withoutPlayer = current.map((team) =>
        team.filter((memberIndex) => memberIndex !== playerIndex),
      );

      if (selectedInTeam) {
        return withoutPlayer;
      }

      return withoutPlayer.map((team, index) =>
        index === teamIndex
          ? [...team, playerIndex].sort((left, right) => left - right)
          : team,
      );
    });
  }

  function handleAddTeam() {
    setTeamSelections((current) => [...current, []]);
  }

  function handleRemoveTeam(teamIndex: number) {
    setTeamSelections((current) =>
      current.filter((_, index) => index !== teamIndex),
    );
  }

  async function handleFullscreenToggle() {
    const gameArea = gameAreaRef.current;

    if (!gameArea) {
      return;
    }

    if (isFullscreen || document.fullscreenElement === gameArea) {
      if (document.fullscreenElement === gameArea) {
        await document.exitFullscreen();
      }
      setIsFullscreen(false);
      return;
    }

    try {
      if (document.fullscreenEnabled) {
        await gameArea.requestFullscreen();
      }
      setIsFullscreen(true);
    } catch {
      setIsFullscreen(true);
    }
  }

  return (
    <main
      className={[
        "public-shell",
        useMobileGameLayout ? "public-shell--mobile-game" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <section className="hero-card">
        <div>
          <span className="eyebrow">Play with bots</span>
          <h1>Smear</h1>
        </div>
        <div className="top-actions">
          <button type="button" className="ghost-button" onClick={onNavigateHome}>
            Home
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateLearn}>
            Learn
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateDonate}>
            Donate
          </button>
          <button
            type="button"
            className="help-button"
            aria-expanded={showHelp}
            aria-controls="smear-help-panel"
            onClick={() => setShowHelp((current) => !current)}
          >
            Help
          </button>
        </div>
      </section>

      {showHelp ? (
        <section
          id="smear-help-panel"
          className="help-panel"
          aria-labelledby="smear-help-heading"
        >
          <div className="help-panel__header">
            <h2 id="smear-help-heading">Table help</h2>
            <button
              type="button"
              className="text-button"
              onClick={() => setShowHelp(false)}
            >
              Close
            </button>
          </div>
          <div className="help-panel__grid">
            <p>
              Set the players and teams on the left, then start a new game when
              every team seat is assigned.
            </p>
            <p>
              When it is your turn, playable cards are active. Cards that cannot
              be played are dimmed.
            </p>
            <p>
              Use full screen for a simpler table view with larger cards and
              fewer distractions.
            </p>
          </div>
        </section>
      ) : null}

      {error ? <div className="banner banner--error">{error}</div> : null}

      <div
        className={[
          "experience-grid",
          setupSidebarOpen
            ? "experience-grid--setup-open"
            : "experience-grid--setup-collapsed",
        ]
          .filter(Boolean)
          .join(" ")}
      >
        <div className="setup-sidebar">
          <button
            type="button"
            className="setup-sidebar__toggle"
            aria-expanded={setupSidebarOpen}
            aria-controls="table-setup-panel"
            onClick={() => setSetupSidebarOpen((current) => !current)}
          >
            {setupSidebarOpen ? "Hide setup" : "New game"}
          </button>

          <aside id="table-setup-panel" className="control-card">
            <div className="card-header">
              <div>
                <span className="eyebrow">Table setup</span>
                <h2>Players and teams</h2>
              </div>
              <button
                type="button"
                className="text-button control-card__collapse"
                onClick={() => setSetupSidebarOpen(false)}
              >
                Hide
              </button>
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
              <div className="team-picker">
                <div className="team-picker__header">
                  <span>Teams</span>
                  <button
                    type="button"
                    className="text-button"
                    onClick={handleAddTeam}
                    disabled={teamSelections.length >= numPlayers}
                  >
                    Add team
                  </button>
                </div>

                {teamSelections.map((team, teamIndex) => (
                  <section key={teamIndex} className="team-picker__team">
                    <div className="team-picker__team-header">
                      <strong>Team {teamIndex + 1}</strong>
                      {teamSelections.length > 1 ? (
                        <button
                          type="button"
                          className="text-button"
                          onClick={() => handleRemoveTeam(teamIndex)}
                        >
                          Remove
                        </button>
                      ) : null}
                    </div>
                    <div className="player-chip-grid">
                      {playerOptions.map((player) => {
                        const isSelected = team.includes(player.index);
                        const isAssignedElsewhere =
                          !isSelected && assignedPlayerIndexes.has(player.index);

                        return (
                          <button
                            key={player.index}
                            type="button"
                            className={[
                              "player-chip",
                              isSelected ? "is-selected" : "",
                              isAssignedElsewhere ? "is-assigned-elsewhere" : "",
                            ]
                              .filter(Boolean)
                              .join(" ")}
                            aria-pressed={isSelected}
                            onClick={() =>
                              handleTeamMemberToggle(teamIndex, player.index)
                            }
                          >
                            {player.name}
                          </button>
                        );
                      })}
                    </div>
                  </section>
                ))}

                {unassignedPlayers.length > 0 ? (
                  <div className="team-picker__unassigned">
                    <span>Unassigned</span>
                    <div className="team-picker__unassigned-list">
                      {unassignedPlayers.map((player) => (
                        <span key={player.index}>{player.name}</span>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}

            <button
              type="button"
              className="cta-button"
              onClick={() => {
                void handleStartTable();
              }}
              disabled={isLoading || !teamsAreComplete}
              title={
                teamsAreComplete
                  ? undefined
                  : "Assign every player before starting a team game"
              }
            >
              New game
            </button>
          </aside>
        </div>

        <section
          ref={gameAreaRef}
          className={[
            "board-column",
            "game-stage",
            isFullscreen ? "game-stage--fullscreen" : "",
          ]
            .filter(Boolean)
            .join(" ")}
        >
          <div className="game-toolbar">
            <div>
              <span className="eyebrow">Game</span>
              <strong>
                {state ? getRoundBanner(state.phase, currentTurnName) : "Ready"}
              </strong>
            </div>
            <div className="game-toolbar__actions">
              {state ? (
                <button
                  type="button"
                  className="ghost-button game-toolbar__mobile-new"
                  onClick={() => {
                    void handleStartTable();
                  }}
                  disabled={isLoading || !teamsAreComplete}
                >
                  New
                </button>
              ) : null}
              <button
                type="button"
                className="ghost-button"
                onClick={() => {
                  void handleFullscreenToggle();
                }}
                disabled={!state || !canUseFullscreenControl}
              >
                {isFullscreen ? "Exit full screen" : "Full screen"}
              </button>
            </div>
          </div>

          {!state ? (
            <section className="empty-state-card">
              <span className="eyebrow">No active table</span>
              <h2>Start a new game from the table setup.</h2>
            </section>
          ) : (
            <>
              <div className="progress-slot">
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
                        style={
                          botProgress
                            ? { width: `${progressPercent}%` }
                            : undefined
                        }
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
              </div>

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

                <div
                  className="seat-grid"
                  data-seat-count={state.round.players.length}
                >
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

function LandingPage({
  onNavigatePlay,
  onNavigateLearn,
  onNavigateDonate,
}: PageNavigationProps) {
  return (
    <main className="public-shell landing-shell">
      <section className="landing-hero">
        <div className="landing-hero__copy">
          <span className="eyebrow">Smear online</span>
          <h1>Smear</h1>
          <p>
            Sit down against bots, or work through one position at a time and
            compare your choice with the best bot.
          </p>
        </div>
        <div className="landing-card-fan" aria-hidden="true">
          {LANDING_CARDS.map((card) => (
            <PlayingCard
              key={card.code}
              card={card}
              compact
              isTrump={card.suit === "H" || card.is_joker}
            />
          ))}
        </div>
      </section>

      <section className="landing-choice-grid" aria-label="Choose how to play">
        <button
          type="button"
          className="landing-choice"
          onClick={onNavigatePlay}
        >
          <span>Play with bots</span>
          <strong>Start or continue a table against computer players.</strong>
        </button>
        <button
          type="button"
          className="landing-choice"
          onClick={onNavigateLearn}
        >
          <span>Learn</span>
          <strong>Practice random bid and card positions, then reveal the bot move.</strong>
        </button>
        <button
          type="button"
          className="landing-choice"
          onClick={onNavigateDonate}
        >
          <span>Support Smear</span>
          <strong>Make a small donation toward hosting and continued tuning.</strong>
        </button>
      </section>
    </main>
  );
}

function DonationPage({
  onNavigateHome,
  onNavigatePlay,
  onNavigateLearn,
}: PageNavigationProps) {
  const client = useMemo(
    () => createApiClient({ apiBaseUrl: API_BASE_URL, sessionId: null }),
    [],
  );
  const [selectedAmountCents, setSelectedAmountCents] = useState(500);
  const [customAmount, setCustomAmount] = useState("");
  const [checkoutError, setCheckoutError] = useState<string | null>(null);
  const [isStartingCheckout, setIsStartingCheckout] = useState(false);
  const donationStatus = getDonationStatus();
  const customAmountCents = customAmount.trim()
    ? parseDonationAmount(customAmount)
    : null;
  const amountCents = customAmount.trim()
    ? customAmountCents
    : selectedAmountCents;
  const amountIsValid = amountCents !== null;

  async function handleDonationSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setCheckoutError(null);

    if (amountCents === null) {
      setCheckoutError(
        `Choose an amount from ${formatDonationAmount(
          DONATION_MIN_CENTS,
        )} to ${formatDonationAmount(DONATION_MAX_CENTS)}.`,
      );
      return;
    }

    setIsStartingCheckout(true);
    try {
      const checkout = await client.createDonationCheckoutSession({
        amount_cents: amountCents,
      });
      window.location.assign(checkout.url);
    } catch (error) {
      setCheckoutError(
        error instanceof Error
          ? error.message
          : "Could not start Stripe Checkout.",
      );
      setIsStartingCheckout(false);
    }
  }

  return (
    <main className="public-shell donation-shell">
      <section className="hero-card">
        <div>
          <span className="eyebrow">Support the site</span>
          <h1>Donate to Smear</h1>
        </div>
        <div className="top-actions">
          <button type="button" className="ghost-button" onClick={onNavigateHome}>
            Home
          </button>
          <button type="button" className="ghost-button" onClick={onNavigatePlay}>
            Play with bots
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateLearn}>
            Learn
          </button>
        </div>
      </section>

      {donationStatus === "success" ? (
        <div className="banner banner--success">
          Thank you for supporting Smear.
        </div>
      ) : null}

      {donationStatus === "cancelled" ? (
        <div className="banner">Donation checkout was cancelled.</div>
      ) : null}

      {checkoutError ? (
        <div className="banner banner--error">{checkoutError}</div>
      ) : null}

      <section className="donation-layout">
        <form className="donation-panel" onSubmit={handleDonationSubmit}>
          <div>
            <span className="eyebrow">Small donation</span>
            <h2>Keep the table open</h2>
            <p>
              A few dollars helps cover hosting, bot experiments, and the
              multiplayer work behind the public Smear table.
            </p>
          </div>

          <div
            className="donation-preset-grid"
            role="group"
            aria-label="Donation amount"
          >
            {DONATION_PRESETS.map((preset) => (
              <button
                key={preset.amountCents}
                type="button"
                className={[
                  "donation-amount-button",
                  !customAmount.trim() &&
                  selectedAmountCents === preset.amountCents
                    ? "is-selected"
                    : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                aria-pressed={
                  !customAmount.trim() &&
                  selectedAmountCents === preset.amountCents
                }
                onClick={() => {
                  setSelectedAmountCents(preset.amountCents);
                  setCustomAmount("");
                  setCheckoutError(null);
                }}
              >
                {formatDonationAmount(preset.amountCents)}
              </button>
            ))}
          </div>

          <label className="field donation-custom-field">
            <span>Custom amount</span>
            <div className="donation-currency-input">
              <span>{DONATION_CURRENCY}</span>
              <input
                type="text"
                inputMode="decimal"
                placeholder="7.50"
                value={customAmount}
                onChange={(event) => {
                  setCustomAmount(event.target.value);
                  setCheckoutError(null);
                }}
              />
            </div>
          </label>

          <button
            type="submit"
            className="cta-button"
            disabled={isStartingCheckout || !amountIsValid}
          >
            {isStartingCheckout
              ? "Opening Stripe"
              : `Donate ${formatDonationAmount(
                  amountCents ?? selectedAmountCents,
                )}`}
          </button>
        </form>

        <aside className="donation-summary" aria-label="Donation details">
          <div>
            <span className="eyebrow">Stripe Checkout</span>
            <h2>Secure payment</h2>
            <p>
              Payments are completed on Stripe. Smear never receives or stores
              card details.
            </p>
          </div>
          <div className="donation-detail-grid">
            <div className="donation-detail">
              <span>Minimum</span>
              <strong>{formatDonationAmount(DONATION_MIN_CENTS)}</strong>
            </div>
            <div className="donation-detail">
              <span>Suggested</span>
              <strong>{formatDonationAmount(500)}</strong>
            </div>
            <div className="donation-detail">
              <span>Maximum</span>
              <strong>{formatDonationAmount(DONATION_MAX_CENTS)}</strong>
            </div>
          </div>
        </aside>
      </section>
    </main>
  );
}

function LearnPage({
  onNavigateHome,
  onNavigatePlay,
  onNavigateDonate,
}: PageNavigationProps) {
  const client = useMemo(
    () => createApiClient({ apiBaseUrl: API_BASE_URL, sessionId: null }),
    [],
  );
  const [challenge, setChallenge] = useState<LearnChallenge | null>(null);
  const [selectedAction, setSelectedAction] = useState<LearnAction | null>(null);
  const [isLoadingChallenge, setIsLoadingChallenge] = useState(true);
  const [learnError, setLearnError] = useState<string | null>(null);

  const loadChallenge = useCallback(async () => {
    setIsLoadingChallenge(true);
    setLearnError(null);
    setSelectedAction(null);
    try {
      const nextChallenge = await client.fetchLearnChallenge();
      setChallenge(nextChallenge);
    } catch (error) {
      setLearnError(
        error instanceof Error
          ? error.message
          : "Could not load a learning position.",
      );
    } finally {
      setIsLoadingChallenge(false);
    }
  }, [client]);

  useEffect(() => {
    void loadChallenge();
  }, [loadChallenge]);

  const actor = challenge
    ? challenge.state.round.players.find(
        (player) => player.name === challenge.actor_name,
      ) ?? null
    : null;
  const selectedKey = selectedAction ? getActionKey(selectedAction) : null;
  const bestKey = challenge ? getActionKey(challenge.best_action) : null;
  const selectedMatchesBest =
    selectedKey !== null && bestKey !== null && selectedKey === bestKey;
  const currentTrick = challenge?.state.round.current_trick ?? null;

  return (
    <main className="public-shell learn-shell">
      <section className="hero-card">
        <div>
          <span className="eyebrow">Learn</span>
          <h1>Practice a position</h1>
        </div>
        <div className="top-actions">
          <button type="button" className="ghost-button" onClick={onNavigateHome}>
            Home
          </button>
          <button type="button" className="ghost-button" onClick={onNavigatePlay}>
            Play with bots
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateDonate}>
            Donate
          </button>
        </div>
      </section>

      {learnError ? (
        <div className="banner banner--error">{learnError}</div>
      ) : null}

      <section className="learn-layout">
        <div className="learn-position">
          <div className="card-header">
            <div>
              <span className="eyebrow">
                {challenge?.phase === "auction" ? "Auction" : "Card play"}
              </span>
              <h2>
                {isLoadingChallenge
                  ? "Loading a position"
                  : challenge?.prompt ?? "No position loaded"}
              </h2>
            </div>
            <button
              type="button"
              className="ghost-button"
              onClick={() => {
                void loadChallenge();
              }}
              disabled={isLoadingChallenge}
            >
              New position
            </button>
          </div>

          {challenge ? (
            <>
              <section className="status-row learn-status-row">
                <div className="status-pill">
                  <span>Trump</span>
                  <strong>{challenge.state.round.trump ?? "-"}</strong>
                </div>
                <div className="status-pill">
                  <span>High bid</span>
                  <strong>{challenge.state.auction.current_high_bid ?? "-"}</strong>
                </div>
                <div className="status-pill">
                  <span>High bidder</span>
                  <strong>{challenge.state.auction.highest_bidder_name ?? "-"}</strong>
                </div>
                <div className="status-pill">
                  <span>Round</span>
                  <strong>{challenge.state.match.round_number}</strong>
                </div>
              </section>

              <section className="learn-table">
                <div className="learn-table__header">
                  <span>Current trick</span>
                  <strong>{currentTrick?.leader_name ?? "-"}</strong>
                </div>
                <div className="learn-trick-row">
                  {currentTrick && currentTrick.plays.length > 0 ? (
                    currentTrick.plays.map((play, index) => (
                      <div
                        key={`${play.player_name}-${play.card.code}-${index}`}
                        className="learn-trick-play"
                      >
                        <span>{play.player_name}</span>
                        <PlayingCard
                          card={play.card}
                          compact
                          isTrump={
                            play.card.is_joker ||
                            play.card.suit === challenge.state.round.trump
                          }
                        />
                      </div>
                    ))
                  ) : (
                    <div className="empty-inline">No cards have been played to this trick.</div>
                  )}
                </div>
              </section>

              <section className="learn-hand">
                <div>
                  <span className="eyebrow">Your hand</span>
                  <h2>{actor?.name ?? challenge.actor_name}</h2>
                </div>
                <div className="hand-grid">
                  {actor ? (
                    sortCardsHighToLow(actor.cards).map((card) => (
                      <PlayingCard
                        key={card.code}
                        card={card}
                        compact
                        isTrump={
                          card.is_joker || card.suit === challenge.state.round.trump
                        }
                        disabled={
                          challenge.phase === "play" &&
                          !challenge.options.some(
                            (option) =>
                              option.type === "play_card" &&
                              option.card_code === card.code,
                          )
                        }
                      />
                    ))
                  ) : (
                    <span className="empty-inline">Hand unavailable.</span>
                  )}
                </div>
              </section>
            </>
          ) : (
            <div className="empty-state-card">
              <span className="eyebrow">Practice</span>
              <h2>{isLoadingChallenge ? "Finding a position." : "Try loading a new position."}</h2>
            </div>
          )}
        </div>

        <aside className="learn-actions">
          <div>
            <span className="eyebrow">Choose</span>
            <h2>Your options</h2>
          </div>

          {challenge ? (
            <div className="learn-option-grid">
              {challenge.options.map((option) => {
                const optionKey = getActionKey(option);
                const optionCard =
                  option.type === "play_card"
                    ? findCardByCode(actor?.cards ?? [], option.card_code)
                    : null;

                return (
                  <button
                    key={optionKey}
                    type="button"
                    className={[
                      "learn-option",
                      selectedKey === optionKey ? "is-selected" : "",
                      selectedAction && bestKey === optionKey ? "is-best" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    onClick={() => setSelectedAction(option)}
                    disabled={Boolean(selectedAction)}
                  >
                    {optionCard ? (
                      <PlayingCard
                        card={optionCard}
                        compact
                        isTrump={
                          optionCard.is_joker ||
                          optionCard.suit === challenge.state.round.trump
                        }
                      />
                    ) : null}
                    <span>{option.label}</span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="empty-inline">Options will appear after the position loads.</div>
          )}

          {challenge && selectedAction ? (
            <section
              className={[
                "learn-result",
                selectedMatchesBest ? "learn-result--match" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              aria-live="polite"
            >
              <span className="eyebrow">Best bot</span>
              <h2>{selectedMatchesBest ? "Same choice" : "Different choice"}</h2>
              <p>
                You chose <strong>{selectedAction.label}</strong>.{" "}
                {challenge.best_bot_label} would choose{" "}
                <strong>{challenge.best_action.label}</strong>.
              </p>
              <button
                type="button"
                className="cta-button"
                onClick={() => {
                  void loadChallenge();
                }}
              >
                Next position
              </button>
            </section>
          ) : null}
        </aside>
      </section>
    </main>
  );
}

export default function App() {
  const [view, setView] = useState<AppView>(() =>
    typeof window === "undefined" ? "home" : viewFromHash(window.location.hash),
  );

  useEffect(() => {
    function handleHashChange() {
      setView(viewFromHash(window.location.hash));
    }

    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, []);

  const navigation = {
    onNavigateHome: () => setHashView("home"),
    onNavigatePlay: () => setHashView("play"),
    onNavigateLearn: () => setHashView("learn"),
    onNavigateDonate: () => setHashView("donate"),
  };

  if (view === "play") {
    return <PlayBotsPage {...navigation} />;
  }

  if (view === "learn") {
    return <LearnPage {...navigation} />;
  }

  if (view === "donate") {
    return <DonationPage {...navigation} />;
  }

  return <LandingPage {...navigation} />;
}
