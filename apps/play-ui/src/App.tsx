import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type TouchEvent,
} from "react";

import {
  buildDisplayCapturedByPlayer,
  createApiClient,
  loadOrCreateSessionId,
  PlayingCard,
  sortCardsHighToLow,
  useSmearGame,
  type Card,
  type LegalAction,
  type LearnAction,
  type LearnChallenge,
  type LobbyState,
  type LobbyStateEvent,
  type Player,
  type ReadyBot,
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
const DONATION_CURRENCY = (
  import.meta.env.VITE_DONATION_CURRENCY ?? "CAD"
).toUpperCase();
const DEFAULT_LEARN_BOT_ID = "optimal-bot";
const MOBILE_TABLE_QUERY =
  "(max-width: 760px), (pointer: coarse) and (max-width: 920px)";
const LOBBY_TOKEN_STORAGE_PREFIX = "smear-lobby-player-token:";

type AppView =
  | "home"
  | "play"
  | "learn"
  | "learn/practice"
  | "learn/tutorial"
  | "donate";
type PlayMode = "menu" | "bots" | "host" | "join" | "lobby";
type TutorialLessonId =
  | "welcome"
  | "deck"
  | "auction"
  | "trump"
  | "legal-plays"
  | "scoring"
  | "match"
  | "ready";

type TutorialAnswer = {
  label: string;
  feedback: string;
  isCorrect: boolean;
};

type TutorialLesson = {
  id: TutorialLessonId;
  eyebrow: string;
  title: string;
  overview: string;
  question: string;
  answers: TutorialAnswer[];
};

const TUTORIAL_LESSONS: TutorialLesson[] = [
  {
    id: "welcome",
    eyebrow: "Lesson 1",
    title: "Meet the table",
    overview:
      "Understand the goal, table size, teams, and what a round looks like.",
    question: "How many tricks are played in one round of Smear?",
    answers: [
      {
        label: "Six, one for every card dealt",
        feedback:
          "Every player receives six cards, and every card is played across six tricks.",
        isCorrect: true,
      },
      {
        label: "Four, one for every player",
        feedback:
          "Tricks are based on cards in each hand, not the number of players. Each hand has six cards.",
        isCorrect: false,
      },
      {
        label: "Until a team reaches 21",
        feedback:
          "Twenty-one ends the match; a single round still contains exactly six tricks.",
        isCorrect: false,
      },
    ],
  },
  {
    id: "deck",
    eyebrow: "Lesson 2",
    title: "Deal and functional deck",
    overview:
      "Learn which cards are used and why a few unseen cards still affect scoring.",
    question:
      "In the usual four-player game, which functional deck is dealt?",
    answers: [
      {
        label: "Ranks 9 through Ace plus two jokers, with two cards hidden",
        feedback:
          "Four hands use 24 cards. The 26-card functional deck leaves two unseen hiding cards.",
        isCorrect: true,
      },
      {
        label: "The full 54-card deck, with 30 cards hidden",
        feedback:
          "Smear removes low ranks to keep the hiding cards close to two; four players use a low of 9.",
        isCorrect: false,
      },
      {
        label: "Ranks 10 through Ace, with no jokers",
        feedback:
          "Jokers are always included in the functional deck and can score points.",
        isCorrect: false,
      },
    ],
  },
  {
    id: "auction",
    eyebrow: "Lesson 3",
    title: "Win the right to lead",
    overview:
      "Bidding is one lap around the table and the winning bid carries risk.",
    question:
      "Nobody has bid before the dealer, who is the last bidder. What can the dealer do?",
    answers: [
      {
        label: "They must bid from 1 to 6",
        feedback:
          "The final bidder cannot pass when no bid has been made, so every round gets an auction winner.",
        isCorrect: true,
      },
      {
        label: "Pass and redeal",
        feedback:
          "There is no all-pass redeal. The last bidder is forced to open the bidding.",
        isCorrect: false,
      },
      {
        label: "Name trump without bidding",
        feedback:
          "Trump is determined later by the first card led, not declared during the auction.",
        isCorrect: false,
      },
    ],
  },
  {
    id: "trump",
    eyebrow: "Lesson 4",
    title: "Trump and winning tricks",
    overview:
      "The auction winner establishes trump with the first lead, then cards compete by priority.",
    question:
      "Trump is hearts. A trick contains the ace of spades, a joker, and the king of hearts. Which wins?",
    answers: [
      {
        label: "King of hearts",
        feedback:
          "Any trump suit card defeats jokers and off-suit cards; the highest played trump wins.",
        isCorrect: true,
      },
      {
        label: "The joker",
        feedback:
          "A joker wins only when no trump suit card appears in the trick.",
        isCorrect: false,
      },
      {
        label: "Ace of spades",
        feedback:
          "An off-suit ace cannot beat a trump card in the trick.",
        isCorrect: false,
      },
    ],
  },
  {
    id: "legal-plays",
    eyebrow: "Lesson 5",
    title: "Choose a legal card",
    overview:
      "Trump and jokers are flexible, but following a non-trump suit and responding to a trump lead impose limits.",
    question:
      "Trump is hearts and diamonds are led. You hold 9D, 10H, J1, and KS. Which cards are legal?",
    answers: [
      {
        label: "9D, 10H, and J1",
        feedback:
          "You may follow diamonds, play trump, or play a joker. You cannot discard KS while you can follow diamonds.",
        isCorrect: true,
      },
      {
        label: "Only 9D",
        feedback:
          "Following suit is legal, but trump cards and jokers are also always legal responses during a trick.",
        isCorrect: false,
      },
      {
        label: "All four cards",
        feedback:
          "KS is an off-suit discard and is illegal while you still hold a diamond to follow the led suit.",
        isCorrect: false,
      },
    ],
  },
  {
    id: "scoring",
    eyebrow: "Lesson 6",
    title: "Score the six points",
    overview:
      "A round awards high, low, jack, two possible joker points, and game.",
    question:
      "You were dealt the lowest visible trump, but an opponent captured it in a trick. Who receives low?",
    answers: [
      {
        label: "You or your team",
        feedback:
          "Low stays with the scoring unit originally dealt and playing that trump card, even if the trick is lost.",
        isCorrect: true,
      },
      {
        label: "The opponent who captured it",
        feedback:
          "Most cards score through capture, but low is the explicit exception.",
        isCorrect: false,
      },
      {
        label: "Nobody",
        feedback:
          "The lowest visible trump is worth one low point whenever it was dealt, regardless of who takes its trick.",
        isCorrect: false,
      },
    ],
  },
  {
    id: "match",
    eyebrow: "Lesson 7",
    title: "Make the bid and win",
    overview:
      "The bidder can reach 21, but failing the contract costs the bid amount.",
    question:
      "A bidding team bids 4 but earns only 3 raw round points. What happens to its match score?",
    answers: [
      {
        label: "It loses 4 match points",
        feedback:
          "Missing the contract sets the bidding unit back by the full bid, not by its earned points.",
        isCorrect: true,
      },
      {
        label: "It adds 3 points",
        feedback:
          "The bidder adds its points only after making its bid.",
        isCorrect: false,
      },
      {
        label: "It loses 1 point",
        feedback:
          "The penalty is not the shortfall. A failed bid subtracts the entire bid amount.",
        isCorrect: false,
      },
    ],
  },
  {
    id: "ready",
    eyebrow: "Lesson 8",
    title: "Play a complete round",
    overview:
      "Put the sequence together before moving from rules to practice or a table.",
    question:
      "Which sequence correctly describes a round from start to finish?",
    answers: [
      {
        label:
          "Bid once each, bidder leads trump, play six tricks, score cards, check the bid",
        feedback:
          "That is the full round loop. You now have the rules needed to begin playing.",
        isCorrect: true,
      },
      {
        label:
          "Choose trump, deal hands, bid repeatedly, score after every trick",
        feedback:
          "Cards are dealt first, the auction is one lap, and trump is established by the bidder's first lead.",
        isCorrect: false,
      },
      {
        label:
          "Deal, play any six tricks, then let the highest score declare trump",
        feedback:
          "Trump must be established on the first lead so it can govern trick play and scoring.",
        isCorrect: false,
      },
    ],
  },
];

const TUTORIAL_CARDS: Record<string, Card> = {
  "10H": { code: "10H", rank: "10", suit: "H", is_joker: false },
  KH: { code: "KH", rank: "K", suit: "H", is_joker: false },
  AD: { code: "AD", rank: "A", suit: "D", is_joker: false },
  AS: { code: "AS", rank: "A", suit: "S", is_joker: false },
  "9D": { code: "9D", rank: "9", suit: "D", is_joker: false },
  KS: { code: "KS", rank: "K", suit: "S", is_joker: false },
  J1: { code: "J1", rank: null, suit: null, is_joker: true },
};

function viewFromHash(hash: string): AppView {
  const route = hash.split("?")[0];

  if (route === "#play") {
    return "play";
  }
  if (route === "#learn") {
    return "learn";
  }
  if (route === "#learn/practice") {
    return "learn/practice";
  }
  if (route === "#learn/tutorial") {
    return "learn/tutorial";
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
  if (amountCents < DONATION_MIN_CENTS || amountCents > DONATION_MAX_CENTS) {
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

function getTurnName(state: LobbyState["game_state"]): string {
  if (!state) {
    return "-";
  }
  if (state.phase === "auction") {
    return state.auction.current_bidder_name;
  }
  if (state.phase === "play") {
    return state.round.current_player_name;
  }
  if (state.phase === "match_complete") {
    return "Match over";
  }
  return "Round over";
}

function normalizeLobbyCode(value: string): string {
  return value.trim().toUpperCase();
}

function getLobbyTokenStorageKey(lobbyCode: string): string {
  return `${LOBBY_TOKEN_STORAGE_PREFIX}${normalizeLobbyCode(lobbyCode)}`;
}

function loadLobbyPlayerToken(lobbyCode: string): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return window.localStorage.getItem(getLobbyTokenStorageKey(lobbyCode));
}

function saveLobbyPlayerToken(lobbyCode: string, playerToken: string) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(getLobbyTokenStorageKey(lobbyCode), playerToken);
}

function getBidActions(actions: LegalAction[]) {
  return actions.filter(
    (action): action is Extract<LegalAction, { type: "bid" }> =>
      action.type === "bid",
  );
}

function getPlayCardActions(actions: LegalAction[]) {
  return actions.filter(
    (action): action is Extract<LegalAction, { type: "play_card" }> =>
      action.type === "play_card",
  );
}

function getVisiblePlayForPlayer(trick: TrickState | null, playerName: string) {
  return trick?.plays.find((play) => play.player_name === playerName) ?? null;
}

function getScoreHeading(roundIsTerminal: boolean): string {
  return roundIsTerminal ? "Round Summary" : "Last Round Summary";
}

function getJokerAwardText(
  results: Array<{ name: string; joker_count: number }>,
) {
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

function SiteFooter() {
  return <footer className="site-footer">By Daniel Markusson</footer>;
}

type BrandLogoProps = {
  className?: string;
};

function BrandLogo({ className = "" }: BrandLogoProps) {
  return (
    <img
      className={["brand-logo", className].filter(Boolean).join(" ")}
      src="/logo.svg"
      alt="Play Smear logo"
    />
  );
}

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
  const setupDrawerTouchStart = useRef<{ x: number; y: number } | null>(null);

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
      : (revealedTrick ?? state.round.current_trick)
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
  const isRoundTerminal = Boolean(state?.round.is_terminal);
  const isMatchComplete = Boolean(state?.match.is_complete);
  const wasMobileLayout = useRef(useMobileGameLayout);
  const isSetupDrawerOpen = setupSidebarOpen && useMobileGameLayout;

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
    if (wasMobileLayout.current && !useMobileGameLayout) {
      setSetupSidebarOpen(false);
    }

    wasMobileLayout.current = useMobileGameLayout;
  }, [useMobileGameLayout]);

  useEffect(() => {
    function handleDrawerEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && isSetupDrawerOpen) {
        setSetupSidebarOpen(false);
      }
    }

    document.addEventListener("keydown", handleDrawerEscape);
    return () => {
      document.removeEventListener("keydown", handleDrawerEscape);
    };
  }, [isSetupDrawerOpen]);

  useEffect(() => {
    document.body.classList.toggle("mobile-game-locked", useMobileGameLayout);

    return () => {
      document.body.classList.remove("mobile-game-locked");
    };
  }, [useMobileGameLayout]);

  function confirmNewGameIfNeeded() {
    if (!state || state.match.is_complete) {
      return true;
    }

    return window.confirm("Start a fresh table with the current setup?");
  }

  async function handleStartTable(): Promise<boolean> {
    if (!confirmNewGameIfNeeded()) {
      return false;
    }

    await handleNewGamePlay();
    return true;
  }

  async function handleStartTableFromPanel() {
    const started = await handleStartTable();
    if (started && useMobileGameLayout) {
      setSetupSidebarOpen(false);
    }
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

  function handleCloseSetupDrawer() {
    setSetupSidebarOpen(false);
  }

  function handleToggleSetupDrawer() {
    setSetupSidebarOpen((current) => !current);
  }

  function handleNavigateHomeFromDrawer() {
    setSetupSidebarOpen(false);
    onNavigateHome();
  }

  function handleDrawerTouchStart(event: TouchEvent<HTMLDivElement>) {
    const touch = event.touches[0];
    if (!touch) {
      return;
    }

    setupDrawerTouchStart.current = {
      x: touch.clientX,
      y: touch.clientY,
    };
  }

  function handleDrawerTouchMove(event: TouchEvent<HTMLDivElement>) {
    const start = setupDrawerTouchStart.current;
    const touch = event.touches[0];
    if (!start || !touch) {
      return;
    }

    const deltaX = touch.clientX - start.x;
    const deltaY = touch.clientY - start.y;
    if (deltaX < -70 && Math.abs(deltaY) < 60) {
      handleCloseSetupDrawer();
      setupDrawerTouchStart.current = null;
    }
  }

  function handleDrawerTouchEnd() {
    setupDrawerTouchStart.current = null;
  }

  const setupPanelContent = (
    <>
      <div className="card-header">
        <div>
          <span className="eyebrow">Table setup</span>
          <h2>Players and teams</h2>
        </div>
        <button
          type="button"
          className="text-button control-card__collapse"
          onClick={handleCloseSetupDrawer}
        >
          Hide
        </button>
      </div>

      <label className="toggle-row">
        <input
          type="checkbox"
          checked={teamsEnabled}
          onChange={(event) => handleTeamsEnabledChange(event.target.checked)}
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
          onChange={(event) => setNumPlayers(Number(event.target.value) || 3)}
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
          void handleStartTableFromPanel();
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
    </>
  );

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
        useMobileGameLayout && isRoundTerminal
          ? "public-shell--round-complete"
          : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <section className="hero-card">
        <div className="hero-card__branding">
          <BrandLogo className="brand-logo--badge" />
          <div>
            <span className="eyebrow">Play with bots</span>
            <h1>Smear</h1>
          </div>
        </div>
        <div className="top-actions">
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigateHome}
          >
            Home
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigateLearn}
          >
            Learn
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigateDonate}
          >
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
            onClick={handleToggleSetupDrawer}
          >
            {setupSidebarOpen ? "Hide setup" : "New game"}
          </button>

          <aside id="table-setup-panel" className="control-card">
            {setupPanelContent}
          </aside>
        </div>

        <section
          ref={gameAreaRef}
          className={[
            "board-column",
            "game-stage",
            isFullscreen ? "game-stage--fullscreen" : "",
            isRoundTerminal ? "game-stage--complete" : "",
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
              {useMobileGameLayout ? (
                <button
                  type="button"
                  className="ghost-button game-toolbar__drawer-toggle"
                  aria-label={
                    isSetupDrawerOpen ? "Close setup menu" : "Open setup menu"
                  }
                  aria-expanded={isSetupDrawerOpen}
                  aria-controls="mobile-setup-drawer"
                  onClick={handleToggleSetupDrawer}
                >
                  Menu
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
                      : (botThinkingName ??
                        getRoundBanner(state.phase, currentTurnName))}
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
                    <h2>
                      {state.phase === "auction" ? "Auction" : "Current trick"}
                    </h2>
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
                    ) : isRoundTerminal && !isMatchComplete ? (
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
                    const tablePlay = getVisiblePlayForPlayer(
                      visibleTrick,
                      player.name,
                    );
                    const displayCaptured = displayCapturedByPlayer?.[
                      player.name
                    ] ?? {
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
                            {state.auction.highest_bidder_name ===
                              player.name &&
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
                        <p className="help-copy">
                          Waiting for {currentTurnName} to act.
                        </p>
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
                        isTrump={
                          !card.is_joker && state.round.trump === card.suit
                        }
                        disabled={
                          state.phase === "play"
                            ? awaitingNextTrick ||
                              isLoading ||
                              !legalCardCodes.has(card.code)
                            : true
                        }
                        onClick={
                          state.phase === "play" &&
                          legalCardCodes.has(card.code)
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
                    {isMatchComplete ? (
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
                        {score.awards.jack.unit_name ??
                          score.awards.jack.reason}
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
                          <strong>
                            {result.match_delta > 0
                              ? `+${result.match_delta}`
                              : result.match_delta}
                          </strong>
                        </div>
                        <p>
                          High {result.breakdown.high} | Jack{" "}
                          {result.breakdown.jack} | Low {result.breakdown.low} |
                          Jokers {result.breakdown.jokers} | Game{" "}
                          {result.breakdown.game}
                        </p>
                      </article>
                    ))}
                  </div>
                </section>
              ) : null}

              {isRoundTerminal ? (
                <section
                  className="mobile-completion-actions"
                  aria-label="Round completion actions"
                >
                  {isMatchComplete ? (
                    <button
                      type="button"
                      className="cta-button"
                      onClick={() => {
                        void handleStartTable();
                      }}
                      disabled={isLoading || !teamsAreComplete}
                    >
                      New game
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="cta-button"
                      onClick={() => {
                        void handleNextRoundPlay();
                      }}
                      disabled={isLoading}
                    >
                      Next round
                    </button>
                  )}
                </section>
              ) : null}
            </>
          )}
        </section>
      </div>

      <SiteFooter />

      <div
        id="mobile-setup-drawer"
        className={["mobile-drawer", isSetupDrawerOpen ? "is-open" : ""]
          .filter(Boolean)
          .join(" ")}
        role="dialog"
        aria-modal="true"
        aria-hidden={!isSetupDrawerOpen}
        aria-label="Table setup"
      >
        <div
          className="mobile-drawer__scrim"
          onClick={handleCloseSetupDrawer}
        />
        <div
          className="mobile-drawer__panel"
          onClick={(event) => event.stopPropagation()}
          onTouchStart={handleDrawerTouchStart}
          onTouchMove={handleDrawerTouchMove}
          onTouchEnd={handleDrawerTouchEnd}
          onTouchCancel={handleDrawerTouchEnd}
        >
          <div className="mobile-drawer__header">
            <div>
              <span className="eyebrow">Menu</span>
              <h2>Table setup</h2>
            </div>
            <button
              type="button"
              className="ghost-button mobile-drawer__close"
              onClick={handleCloseSetupDrawer}
            >
              Close
            </button>
          </div>
          <nav className="mobile-drawer__nav" aria-label="Quick navigation">
            <button
              type="button"
              className="ghost-button"
              onClick={handleNavigateHomeFromDrawer}
            >
              Home
            </button>
          </nav>
          <div className="mobile-drawer__content">
            <aside className="control-card">{setupPanelContent}</aside>
          </div>
        </div>
      </div>
    </main>
  );
}

type PlayPageProps = PageNavigationProps;

type PlayModeMenuProps = PageNavigationProps & {
  onChooseBots: () => void;
  onChooseHost: () => void;
  onChooseJoin: () => void;
};

function PlayModeMenu({
  onNavigateHome,
  onNavigateLearn,
  onNavigateDonate,
  onChooseBots,
  onChooseHost,
  onChooseJoin,
}: PlayModeMenuProps) {
  return (
    <main className="public-shell landing-shell">
      <section className="hero-card">
        <div className="hero-card__branding">
          <BrandLogo className="brand-logo--badge" />
          <div>
            <span className="eyebrow">Play</span>
            <h1>Choose a table</h1>
          </div>
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
        </div>
      </section>

      <section className="landing-choice-grid" aria-label="Play options">
        <button type="button" className="landing-choice" onClick={onChooseBots}>
          <span>Play with bots</span>
          <strong>Start or continue a local table against computer players.</strong>
        </button>
        <button type="button" className="landing-choice" onClick={onChooseHost}>
          <span>Start a lobby</span>
          <strong>Create a lobby code, fill the seats, then start a real table.</strong>
        </button>
        <button type="button" className="landing-choice" onClick={onChooseJoin}>
          <span>Enter lobby code</span>
          <strong>Join a friend’s table from this device.</strong>
        </button>
      </section>

      <SiteFooter />
    </main>
  );
}

type HostLobbyPageProps = PageNavigationProps & {
  onBackToPlayMenu: () => void;
  onOpenLobby: (lobby: LobbyState) => void;
};

function HostLobbyPage({
  onBackToPlayMenu,
  onNavigateHome,
  onNavigateLearn,
  onNavigateDonate,
  onOpenLobby,
}: HostLobbyPageProps) {
  const client = useMemo(
    () => createApiClient({ apiBaseUrl: API_BASE_URL, sessionId: null }),
    [],
  );
  const [hostName, setHostName] = useState("You");
  const [numPlayers, setNumPlayers] = useState(4);
  const [teamsEnabled, setTeamsEnabled] = useState(true);
  const [teamSelections, setTeamSelections] = useState<number[][]>(
    DEFAULT_TEAM_SELECTIONS,
  );
  const [isCreatingLobby, setIsCreatingLobby] = useState(false);
  const [lobbyError, setLobbyError] = useState<string | null>(null);

  const seatOptions = useMemo(
    () =>
      Array.from({ length: numPlayers }, (_, index) => ({
        index,
        name: index === 0 ? hostName.trim() || "You" : `Seat ${index + 1}`,
      })),
    [hostName, numPlayers],
  );
  const assignedSeatIndexes = useMemo(
    () => new Set(teamSelections.flat()),
    [teamSelections],
  );
  const unassignedSeats = seatOptions.filter(
    (seat) => !assignedSeatIndexes.has(seat.index),
  );
  const teamsAreComplete = !teamsEnabled || unassignedSeats.length === 0;

  useEffect(() => {
    setTeamSelections((current) => {
      const filtered = current
        .map((team) => team.filter((seatIndex) => seatIndex < numPlayers))
        .filter((team) => team.length > 0);
      return filtered.length > 0 ? filtered : [[]];
    });
  }, [numPlayers]);

  function handleTeamMemberToggle(teamIndex: number, seatIndex: number) {
    setTeamSelections((current) => {
      const selectedInTeam = current[teamIndex]?.includes(seatIndex) ?? false;
      const withoutSeat = current.map((team) =>
        team.filter((memberIndex) => memberIndex !== seatIndex),
      );

      if (selectedInTeam) {
        return withoutSeat;
      }

      return withoutSeat.map((team, index) =>
        index === teamIndex
          ? [...team, seatIndex].sort((left, right) => left - right)
          : team,
      );
    });
  }

  async function handleCreateLobby(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsCreatingLobby(true);
    setLobbyError(null);
    try {
      const lobby = await client.createLobby({
        host_name: hostName,
        num_players: numPlayers,
        teams: teamsEnabled ? teamSelections : null,
        host_seat_index: 0,
      });
      if (lobby.you) {
        saveLobbyPlayerToken(lobby.code, lobby.you.player_token);
      }
      onOpenLobby(lobby);
    } catch (error) {
      setLobbyError(
        error instanceof Error ? error.message : "Could not create lobby.",
      );
    } finally {
      setIsCreatingLobby(false);
    }
  }

  return (
    <main className="public-shell lobby-shell">
      <section className="hero-card">
        <div className="hero-card__branding">
          <BrandLogo className="brand-logo--badge" />
          <div>
            <span className="eyebrow">Start a lobby</span>
            <h1>Set up the table</h1>
          </div>
        </div>
        <div className="top-actions">
          <button type="button" className="ghost-button" onClick={onBackToPlayMenu}>
            Play
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateHome}>
            Home
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateLearn}>
            Learn
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateDonate}>
            Donate
          </button>
        </div>
      </section>

      {lobbyError ? <div className="banner banner--error">{lobbyError}</div> : null}

      <form className="lobby-setup-grid" onSubmit={handleCreateLobby}>
        <section className="control-card lobby-panel">
          <div className="card-header">
            <div>
              <span className="eyebrow">Seats</span>
              <h2>Lobby details</h2>
            </div>
          </div>

          <label className="field">
            <span>Your name</span>
            <input
              type="text"
              value={hostName}
              maxLength={32}
              onChange={(event) => setHostName(event.target.value)}
            />
          </label>

          <label className="field">
            <span>Seats</span>
            <input
              type="number"
              min={3}
              max={8}
              value={numPlayers}
              onChange={(event) => setNumPlayers(Number(event.target.value) || 3)}
            />
          </label>

          <label className="toggle-row">
            <input
              type="checkbox"
              checked={teamsEnabled}
              onChange={(event) => setTeamsEnabled(event.target.checked)}
            />
            <span>Play with teams</span>
          </label>

          <button
            type="submit"
            className="cta-button"
            disabled={isCreatingLobby || !teamsAreComplete}
          >
            {isCreatingLobby ? "Creating lobby" : "Create lobby"}
          </button>
        </section>

        {teamsEnabled ? (
          <section className="control-card lobby-panel">
            <div className="team-picker">
              <div className="team-picker__header">
                <span>Teams</span>
                <button
                  type="button"
                  className="text-button"
                  onClick={() => setTeamSelections((current) => [...current, []])}
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
                        onClick={() =>
                          setTeamSelections((current) =>
                            current.filter((_, index) => index !== teamIndex),
                          )
                        }
                      >
                        Remove
                      </button>
                    ) : null}
                  </div>
                  <div className="player-chip-grid">
                    {seatOptions.map((seat) => {
                      const isSelected = team.includes(seat.index);
                      const isAssignedElsewhere =
                        !isSelected && assignedSeatIndexes.has(seat.index);

                      return (
                        <button
                          key={seat.index}
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
                            handleTeamMemberToggle(teamIndex, seat.index)
                          }
                        >
                          {seat.name}
                        </button>
                      );
                    })}
                  </div>
                </section>
              ))}

              {unassignedSeats.length > 0 ? (
                <div className="team-picker__unassigned">
                  <span>Unassigned</span>
                  <div className="team-picker__unassigned-list">
                    {unassignedSeats.map((seat) => (
                      <span key={seat.index}>{seat.name}</span>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        ) : null}
      </form>

      <SiteFooter />
    </main>
  );
}

type JoinLobbyPageProps = PageNavigationProps & {
  onBackToPlayMenu: () => void;
  onOpenLobby: (lobby: LobbyState) => void;
};

function JoinLobbyPage({
  onBackToPlayMenu,
  onNavigateHome,
  onNavigateLearn,
  onNavigateDonate,
  onOpenLobby,
}: JoinLobbyPageProps) {
  const client = useMemo(
    () => createApiClient({ apiBaseUrl: API_BASE_URL, sessionId: null }),
    [],
  );
  const [lobbyCode, setLobbyCode] = useState("");
  const [playerName, setPlayerName] = useState("");
  const [foundLobby, setFoundLobby] = useState<LobbyState | null>(null);
  const [joinError, setJoinError] = useState<string | null>(null);
  const [isLoadingLobby, setIsLoadingLobby] = useState(false);

  async function handleFindLobby(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedCode = normalizeLobbyCode(lobbyCode);
    if (!normalizedCode) {
      setJoinError("Enter a lobby code.");
      return;
    }

    setIsLoadingLobby(true);
    setJoinError(null);
    try {
      const storedToken = loadLobbyPlayerToken(normalizedCode);
      const lobby = await client.fetchLobby(normalizedCode, storedToken);
      if (lobby.you) {
        onOpenLobby(lobby);
        return;
      }
      setFoundLobby(lobby);
    } catch (error) {
      setFoundLobby(null);
      setJoinError(
        error instanceof Error ? error.message : "Could not find that lobby.",
      );
    } finally {
      setIsLoadingLobby(false);
    }
  }

  async function handleJoinSeat(seatIndex: number) {
    const normalizedCode = normalizeLobbyCode(lobbyCode);
    setIsLoadingLobby(true);
    setJoinError(null);
    try {
      const lobby = await client.joinLobby(normalizedCode, {
        player_name: playerName,
        seat_index: seatIndex,
      });
      if (lobby.you) {
        saveLobbyPlayerToken(lobby.code, lobby.you.player_token);
      }
      onOpenLobby(lobby);
    } catch (error) {
      setJoinError(
        error instanceof Error ? error.message : "Could not join that lobby.",
      );
    } finally {
      setIsLoadingLobby(false);
    }
  }

  return (
    <main className="public-shell lobby-shell">
      <section className="hero-card">
        <div className="hero-card__branding">
          <BrandLogo className="brand-logo--badge" />
          <div>
            <span className="eyebrow">Enter lobby code</span>
            <h1>Join a table</h1>
          </div>
        </div>
        <div className="top-actions">
          <button type="button" className="ghost-button" onClick={onBackToPlayMenu}>
            Play
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateHome}>
            Home
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateLearn}>
            Learn
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateDonate}>
            Donate
          </button>
        </div>
      </section>

      {joinError ? <div className="banner banner--error">{joinError}</div> : null}

      <section className="lobby-setup-grid">
        <form className="control-card lobby-panel" onSubmit={handleFindLobby}>
          <div className="card-header">
            <div>
              <span className="eyebrow">Code</span>
              <h2>Find lobby</h2>
            </div>
          </div>

          <label className="field">
            <span>Lobby code</span>
            <input
              type="text"
              value={lobbyCode}
              autoCapitalize="characters"
              maxLength={16}
              onChange={(event) => {
                setLobbyCode(normalizeLobbyCode(event.target.value));
                setFoundLobby(null);
              }}
            />
          </label>

          <label className="field">
            <span>Your name</span>
            <input
              type="text"
              value={playerName}
              maxLength={32}
              onChange={(event) => setPlayerName(event.target.value)}
            />
          </label>

          <button
            type="submit"
            className="cta-button"
            disabled={isLoadingLobby || !normalizeLobbyCode(lobbyCode)}
          >
            {isLoadingLobby ? "Finding lobby" : "Find lobby"}
          </button>
        </form>

        {foundLobby ? (
          <section className="control-card lobby-panel">
            <div className="card-header">
              <div>
                <span className="eyebrow">Lobby {foundLobby.code}</span>
                <h2>Choose a seat</h2>
              </div>
            </div>

            {foundLobby.status !== "waiting" ? (
              <div className="empty-inline">This lobby has already started.</div>
            ) : (
              <div className="lobby-seat-list">
                {foundLobby.seats.map((seat) => (
                  <button
                    key={seat.index}
                    type="button"
                    className="lobby-seat-button"
                    disabled={
                      isLoadingLobby || seat.is_occupied || !playerName.trim()
                    }
                    onClick={() => {
                      void handleJoinSeat(seat.index);
                    }}
                  >
                    <span>Seat {seat.index + 1}</span>
                    <strong>
                      {seat.player_name ?? "Open"}
                      {seat.bot_label ? ` (${seat.bot_label})` : ""}
                    </strong>
                  </button>
                ))}
              </div>
            )}
          </section>
        ) : null}
      </section>

      <SiteFooter />
    </main>
  );
}

type MultiplayerLobbyPageProps = PageNavigationProps & {
  initialLobby: LobbyState | null;
  lobbyCode: string;
  playerToken: string;
  onBackToPlayMenu: () => void;
};

function MultiplayerLobbyPage({
  initialLobby,
  lobbyCode,
  playerToken,
  onBackToPlayMenu,
  onNavigateHome,
  onNavigateLearn,
  onNavigateDonate,
}: MultiplayerLobbyPageProps) {
  const client = useMemo(
    () => createApiClient({ apiBaseUrl: API_BASE_URL, sessionId: null }),
    [],
  );
  const [lobby, setLobby] = useState<LobbyState | null>(initialLobby);
  const [lobbyError, setLobbyError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [availableBots, setAvailableBots] = useState<ReadyBot[]>([]);
  const [botSelections, setBotSelections] = useState<Record<number, string>>({});

  const loadLobby = useCallback(async () => {
    try {
      const nextLobby = await client.fetchLobby(lobbyCode, playerToken);
      setLobby(nextLobby);
      setLobbyError(null);
    } catch (error) {
      setLobbyError(
        error instanceof Error ? error.message : "Could not load lobby.",
      );
    }
  }, [client, lobbyCode, playerToken]);

  useEffect(() => {
    void loadLobby();
  }, [loadLobby]);

  useEffect(() => {
    let isActive = true;

    void client
      .fetchBots()
      .then((payload) => {
        if (isActive) {
          setAvailableBots(payload.bots);
        }
      })
      .catch((error) => {
        if (isActive) {
          setLobbyError(
            error instanceof Error ? error.message : "Could not load bots.",
          );
        }
      });

    return () => {
      isActive = false;
    };
  }, [client]);

  useEffect(() => {
    let isClosed = false;
    let reconnectTimeoutId: number | null = null;
    let reconnectDelayMs = 500;
    let socket: WebSocket | null = null;

    function scheduleReconnect() {
      if (isClosed || reconnectTimeoutId !== null) {
        return;
      }
      reconnectTimeoutId = window.setTimeout(() => {
        reconnectTimeoutId = null;
        connect();
      }, reconnectDelayMs);
      reconnectDelayMs = Math.min(reconnectDelayMs * 2, 5000);
    }

    function connect() {
      socket = client.openLobbyEvents(lobbyCode, playerToken);
      if (socket === null) {
        return;
      }
      socket.onopen = () => {
        reconnectDelayMs = 500;
        void loadLobby();
      };
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(String(event.data)) as LobbyStateEvent;
          if (message.type === "lobby_state") {
            setLobby(message.lobby);
            setLobbyError(null);
          }
        } catch {
          setLobbyError("Received an invalid lobby update.");
        }
      };
      socket.onclose = () => {
        socket = null;
        scheduleReconnect();
      };
      socket.onerror = () => {
        socket?.close();
      };
    }

    connect();

    return () => {
      isClosed = true;
      if (reconnectTimeoutId !== null) {
        window.clearTimeout(reconnectTimeoutId);
      }
      socket?.close();
    };
  }, [client, loadLobby, lobbyCode, playerToken]);

  useEffect(() => {
    function handleVisibilityChange() {
      if (document.visibilityState === "visible") {
        void loadLobby();
      }
    }

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [loadLobby]);

  async function runLobbyTask(task: () => Promise<LobbyState>) {
    setIsLoading(true);
    setLobbyError(null);
    try {
      setLobby(await task());
    } catch (error) {
      setLobbyError(
        error instanceof Error ? error.message : "Lobby action failed.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  const state = lobby?.game_state ?? null;
  const currentTurnName = getTurnName(state);
  const isYourTurn =
    Boolean(lobby?.you?.player_name) && lobby?.you?.player_name === currentTurnName;
  const bidActions = getBidActions(lobby?.legal_actions ?? []);
  const playActions = getPlayCardActions(lobby?.legal_actions ?? []);
  const canPassAuction =
    lobby?.legal_actions.some((action) => action.type === "pass") ?? false;
  const legalCardCodes = new Set(playActions.map((action) => action.card_code));
  const youPlayer = state?.round.players.find(
    (player) => player.name === lobby?.you?.player_name,
  ) ?? null;
  const orderedHand = youPlayer ? sortCardsHighToLow(youPlayer.cards) : [];
  const visibleTrick = state
    ? state.round.current_trick.plays.length > 0
      ? state.round.current_trick
      : state.round.current_trick
    : null;
  const completedTricks = state ? [...state.round.trick_history].reverse() : [];
  const displayCapturedByPlayer = state
    ? buildDisplayCapturedByPlayer(state.round)
    : null;
  const defaultLobbyBotId =
    availableBots.find((bot) => bot.id === "greedy")?.id ??
    availableBots[0]?.id ??
    "";

  function getSelectedBotId(seatIndex: number): string {
    return botSelections[seatIndex] ?? defaultLobbyBotId;
  }

  async function handleCopyLobbyCode() {
    try {
      await navigator.clipboard?.writeText(lobby?.code ?? lobbyCode);
    } catch {
      setLobbyError("Could not copy the lobby code on this device.");
    }
  }

  function handleBotSelectionChange(seatIndex: number, botId: string) {
    setBotSelections((current) => ({
      ...current,
      [seatIndex]: botId,
    }));
  }

  async function handleAddBotToSeat(seatIndex: number) {
    if (!lobby) {
      return;
    }

    const botId = getSelectedBotId(seatIndex);
    if (!botId) {
      setLobbyError("No bot is available to add.");
      return;
    }

    await runLobbyTask(() =>
      client.addLobbyBot(lobby.code, {
        player_token: playerToken,
        seat_index: seatIndex,
        bot_id: botId,
      }),
    );
  }

  async function handleRemoveBotFromSeat(seatIndex: number) {
    if (!lobby) {
      return;
    }

    await runLobbyTask(() =>
      client.removeLobbyBot(lobby.code, {
        player_token: playerToken,
        seat_index: seatIndex,
      }),
    );
  }

  return (
    <main className="public-shell lobby-shell">
      <section className="hero-card">
        <div className="hero-card__branding">
          <BrandLogo className="brand-logo--badge" />
          <div>
            <span className="eyebrow">Lobby {lobby?.code ?? lobbyCode}</span>
            <h1>{state ? "Smear" : "Waiting room"}</h1>
          </div>
        </div>
        <div className="top-actions">
          <button type="button" className="ghost-button" onClick={onBackToPlayMenu}>
            Play
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateHome}>
            Home
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateLearn}>
            Learn
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateDonate}>
            Donate
          </button>
        </div>
      </section>

      {lobbyError ? <div className="banner banner--error">{lobbyError}</div> : null}

      {!lobby ? (
        <section className="empty-state-card">
          <span className="eyebrow">Lobby</span>
          <h2>Loading lobby.</h2>
        </section>
      ) : lobby.status === "waiting" ? (
        <section className="lobby-room-grid">
          <div className="control-card lobby-panel">
            <div className="card-header">
              <div>
                <span className="eyebrow">Code</span>
                <h2 className="lobby-code">{lobby.code}</h2>
              </div>
              <button
                type="button"
                className="ghost-button"
                onClick={() => {
                  void handleCopyLobbyCode();
                }}
              >
                Copy
              </button>
            </div>
            <div className="lobby-seat-list">
              {lobby.seats.map((seat) => (
                <div key={seat.index} className="lobby-seat-row">
                  <span>Seat {seat.index + 1}</span>
                  <div className="lobby-seat-row__body">
                    <strong>{seat.player_name ?? "Open"}</strong>
                    {seat.bot_label ? <small>{seat.bot_label}</small> : null}
                  </div>
                  <div className="lobby-seat-row__badges">
                    {seat.is_host ? <em>Host</em> : null}
                    {seat.is_bot ? <em>Bot</em> : null}
                  </div>
                  {lobby.you?.is_host ? (
                    <div className="lobby-seat-controls">
                      {!seat.is_occupied ? (
                        <>
                          <label className="field lobby-seat-controls__select">
                            <span>Bot</span>
                            <select
                              value={getSelectedBotId(seat.index)}
                              onChange={(event) =>
                                handleBotSelectionChange(
                                  seat.index,
                                  event.target.value,
                                )
                              }
                              disabled={availableBots.length === 0 || isLoading}
                            >
                              {availableBots.map((bot) => (
                                <option key={bot.id} value={bot.id}>
                                  {bot.label}
                                </option>
                              ))}
                            </select>
                          </label>
                          <button
                            type="button"
                            className="ghost-button"
                            disabled={
                              isLoading ||
                              availableBots.length === 0 ||
                              !getSelectedBotId(seat.index)
                            }
                            onClick={() => {
                              void handleAddBotToSeat(seat.index);
                            }}
                          >
                            Add bot
                          </button>
                        </>
                      ) : seat.is_bot ? (
                        <button
                          type="button"
                          className="ghost-button"
                          disabled={isLoading}
                          onClick={() => {
                            void handleRemoveBotFromSeat(seat.index);
                          }}
                        >
                          Remove bot
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>

          <aside className="control-card lobby-panel">
            <div>
              <span className="eyebrow">Start</span>
              <h2>{lobby.is_full ? "Ready to play" : "Waiting for seats"}</h2>
              <p className="help-copy">
                Share the lobby code with the other players. The host can start
                once every seat is filled.
              </p>
            </div>
            <button
              type="button"
              className="cta-button"
              disabled={!lobby.you?.is_host || !lobby.is_full || isLoading}
              onClick={() => {
                void runLobbyTask(() =>
                  client.startLobby(lobby.code, playerToken),
                );
              }}
            >
              {isLoading ? "Starting" : "Start game"}
            </button>
          </aside>
        </section>
      ) : state ? (
        <>
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
              <strong>{getRoundBanner(state.phase, currentTurnName)}</strong>
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
                {state.round.is_terminal && !state.match.is_complete ? (
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => {
                      void runLobbyTask(() =>
                        client.nextLobbyRound(lobby.code, playerToken),
                      );
                    }}
                    disabled={isLoading}
                  >
                    Next round
                  </button>
                ) : null}
              </div>
            </div>

            <div className="seat-grid" data-seat-count={state.round.players.length}>
              {state.round.players.map((player: Player) => {
                const tablePlay = getVisiblePlayForPlayer(visibleTrick, player.name);
                const displayCaptured = displayCapturedByPlayer?.[player.name] ?? {
                  cards: player.captured_cards,
                  count: player.captured_count,
                };
                const isCurrentPlayer = player.name === currentTurnName;

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
                          {player.name === lobby.you?.player_name
                            ? "You"
                            : (player.bot_label ?? "Human")}
                        </p>
                      </div>
                      <span className="seat-card__meta">
                        {player.card_count} cards
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
                  {isYourTurn ? (
                    <div className="button-row">
                      {bidActions.map((action) => (
                        <button
                          key={action.amount}
                          type="button"
                          disabled={isLoading}
                          onClick={() => {
                            void runLobbyTask(() =>
                              client.placeLobbyBid(
                                lobby.code,
                                playerToken,
                                action.amount,
                              ),
                            );
                          }}
                        >
                          Bid {action.amount}
                        </button>
                      ))}
                      {canPassAuction ? (
                        <button
                          type="button"
                          disabled={isLoading}
                          onClick={() => {
                            void runLobbyTask(() =>
                              client.passLobbyAuction(lobby.code, playerToken),
                            );
                          }}
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
                <h2>{lobby.you?.player_name ?? "Player"}</h2>
              </div>
            </div>

            {orderedHand.length > 0 ? (
              <div className="hand-grid">
                {orderedHand.map((card) => (
                  <PlayingCard
                    key={card.code}
                    card={card}
                    isTrump={!card.is_joker && state.round.trump === card.suit}
                    disabled={
                      state.phase !== "play" ||
                      !isYourTurn ||
                      isLoading ||
                      !legalCardCodes.has(card.code)
                    }
                    onClick={
                      state.phase === "play" &&
                      isYourTurn &&
                      legalCardCodes.has(card.code)
                        ? () => {
                            void runLobbyTask(() =>
                              client.playLobbyCard(
                                lobby.code,
                                playerToken,
                                card.code,
                              ),
                            );
                          }
                        : undefined
                    }
                  />
                ))}
              </div>
            ) : (
              <div className="empty-inline">
                {state.round.is_terminal
                  ? "The round is complete."
                  : "Your hand will appear after the game starts."}
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
              </div>
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
            </section>
          ) : null}

          {lobby.score ? (
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
                  <strong>{lobby.score.awards.high.unit_name}</strong>
                </div>
                <div className="award-card">
                  <span>Jack</span>
                  <strong>
                    {lobby.score.awards.jack.unit_name ??
                      lobby.score.awards.jack.reason}
                  </strong>
                </div>
                <div className="award-card">
                  <span>Low</span>
                  <strong>{lobby.score.awards.low.unit_name}</strong>
                </div>
                <div className="award-card">
                  <span>Jokers</span>
                  <strong>{getJokerAwardText(lobby.score.results)}</strong>
                </div>
                <div className="award-card">
                  <span>Game</span>
                  <strong>
                    {lobby.score.awards.game.unit_name ??
                      lobby.score.awards.game.tied_unit_names?.join(", ") ??
                      "-"}
                  </strong>
                </div>
              </div>
            </section>
          ) : null}
        </>
      ) : (
        <section className="empty-state-card">
          <span className="eyebrow">Lobby</span>
          <h2>The game has not started yet.</h2>
        </section>
      )}

      <SiteFooter />
    </main>
  );
}

function PlayPage(navigation: PlayPageProps) {
  const [mode, setMode] = useState<PlayMode>("menu");
  const [activeLobbyCode, setActiveLobbyCode] = useState<string | null>(null);
  const [activePlayerToken, setActivePlayerToken] = useState<string | null>(null);
  const [initialLobby, setInitialLobby] = useState<LobbyState | null>(null);

  function handleOpenLobby(lobby: LobbyState) {
    if (!lobby.you) {
      return;
    }
    saveLobbyPlayerToken(lobby.code, lobby.you.player_token);
    setActiveLobbyCode(lobby.code);
    setActivePlayerToken(lobby.you.player_token);
    setInitialLobby(lobby);
    setMode("lobby");
  }

  function handleBackToPlayMenu() {
    setMode("menu");
  }

  if (mode === "bots") {
    return <PlayBotsPage {...navigation} />;
  }

  if (mode === "host") {
    return (
      <HostLobbyPage
        {...navigation}
        onBackToPlayMenu={handleBackToPlayMenu}
        onOpenLobby={handleOpenLobby}
      />
    );
  }

  if (mode === "join") {
    return (
      <JoinLobbyPage
        {...navigation}
        onBackToPlayMenu={handleBackToPlayMenu}
        onOpenLobby={handleOpenLobby}
      />
    );
  }

  if (mode === "lobby" && activeLobbyCode && activePlayerToken) {
    return (
      <MultiplayerLobbyPage
        {...navigation}
        initialLobby={initialLobby}
        lobbyCode={activeLobbyCode}
        playerToken={activePlayerToken}
        onBackToPlayMenu={handleBackToPlayMenu}
      />
    );
  }

  return (
    <PlayModeMenu
      {...navigation}
      onChooseBots={() => setMode("bots")}
      onChooseHost={() => setMode("host")}
      onChooseJoin={() => setMode("join")}
    />
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
          <div className="landing-hero__branding">
            <BrandLogo className="brand-logo--hero" />
            <div>
              <h1>Smear online</h1>
            </div>
          </div>
          <p>
            Sit down with friends, play a fast table against bots, learn the
            rules from the beginning, or practice positions against the best
            bot.
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
          <span>Play</span>
          <strong>Choose bots, host a lobby, or enter a lobby code.</strong>
        </button>
        <button
          type="button"
          className="landing-choice"
          onClick={onNavigateLearn}
        >
          <span>Learn</span>
          <strong>
            Take the complete tutorial or practice bid and card positions.
          </strong>
        </button>
        <button
          type="button"
          className="landing-choice"
          onClick={onNavigateDonate}
        >
          <span>Support Smear</span>
          <strong>
            Make a small donation toward hosting and continued tuning.
          </strong>
        </button>
      </section>

      <p className="landing-note">
        The best bot to play against is the Optimal bot, which is designed to
        play like a human and has no information about human hands. All bots
        except the omniscient bots have the same information a human would have.
        Questions? Reach out at{" "}
        <a href="mailto:support@play-smear.com">support@play-smear.com</a>.
      </p>

      <SiteFooter />
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
        <div className="hero-card__branding">
          <BrandLogo className="brand-logo--badge" />
          <div>
            <span className="eyebrow">Support the site</span>
            <h1>Donate to Smear</h1>
          </div>
        </div>
        <div className="top-actions">
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigateHome}
          >
            Home
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigatePlay}
          >
            Play
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigateLearn}
          >
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

      <SiteFooter />
    </main>
  );
}

type LearnPageProps = PageNavigationProps & {
  onChoosePractice: () => void;
  onChooseTutorial: () => void;
};

function LearnPage({
  onNavigateHome,
  onNavigatePlay,
  onNavigateDonate,
  onChoosePractice,
  onChooseTutorial,
}: LearnPageProps) {
  return (
    <main className="public-shell learn-menu-shell">
      <section className="hero-card">
        <div className="hero-card__branding">
          <BrandLogo className="brand-logo--badge" />
          <div>
            <span className="eyebrow">Learn</span>
            <h1>Learn Smear</h1>
          </div>
        </div>
        <div className="top-actions">
          <button type="button" className="ghost-button" onClick={onNavigateHome}>
            Home
          </button>
          <button type="button" className="ghost-button" onClick={onNavigatePlay}>
            Play
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigateDonate}
          >
            Donate
          </button>
        </div>
      </section>

      <section className="learn-menu-intro">
        <span className="eyebrow">Choose a path</span>
        <h2>Start with the rules or test your decisions</h2>
        <p>
          Tutorial is designed for a first-time player. Practice drops you into
          auction and card-play positions once you understand the table.
        </p>
      </section>

      <section className="learn-mode-grid" aria-label="Learning modes">
        <button
          type="button"
          className="learn-mode-card learn-mode-card--featured"
          onClick={onChooseTutorial}
        >
          <span className="eyebrow">Start here</span>
          <h2>Tutorial</h2>
          <p>
            Learn the deck, bidding, trump, legal plays, scoring, and how to
            win a match in eight guided lessons.
          </p>
          <strong>Begin tutorial</strong>
        </button>
        <button
          type="button"
          className="learn-mode-card"
          onClick={onChoosePractice}
        >
          <span className="eyebrow">Already know the rules?</span>
          <h2>Practice</h2>
          <p>
            Make a bid or play a card in a generated position, then compare
            your decision with a selected bot.
          </p>
          <strong>Practice positions</strong>
        </button>
      </section>

      <SiteFooter />
    </main>
  );
}

type TutorialCardPlayProps = {
  cardCode: string;
  label: string;
  isTrump?: boolean;
  disabled?: boolean;
};

function TutorialCardPlay({
  cardCode,
  label,
  isTrump = false,
  disabled = false,
}: TutorialCardPlayProps) {
  return (
    <div className="tutorial-card-play">
      <span>{label}</span>
      <PlayingCard
        card={TUTORIAL_CARDS[cardCode]}
        compact
        isTrump={isTrump}
        disabled={disabled}
      />
    </div>
  );
}

function TutorialLessonContent({ lessonId }: { lessonId: TutorialLessonId }) {
  if (lessonId === "welcome") {
    return (
      <>
        <p className="tutorial-lead">
          Smear is a trick-taking card game for 3 to 8 players. Four players on
          two teams is the most natural table: teammates combine their round
          points and match score.
        </p>
        <div className="tutorial-rule-grid">
          <article>
            <strong>Goal</strong>
            <span>Reach 21 match points by winning a bid and scoring a round.</span>
          </article>
          <article>
            <strong>Hand</strong>
            <span>Each player receives 6 cards every round.</span>
          </article>
          <article>
            <strong>Round</strong>
            <span>An auction followed by 6 tricks, using every card in hand.</span>
          </article>
          <article>
            <strong>Teams</strong>
            <span>Play individually or in any configured team grouping.</span>
          </article>
        </div>
        <div className="tutorial-callout">
          <strong>What is a trick?</strong>
          <p>
            Each player contributes one card. One card wins, and its player
            captures the cards for scoring. The trick winner leads the next
            trick.
          </p>
        </div>
      </>
    );
  }

  if (lessonId === "deck") {
    return (
      <>
        <p className="tutorial-lead">
          The full deck has 52 standard cards and 2 jokers. Smear deals from a
          smaller functional deck each round so there are only a few unknown
          hiding cards left after every player receives six cards.
        </p>
        <p>
          Ranks run from low to high: <strong>2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A</strong>.
          The deck is shuffled fresh each round.
        </p>
        <div className="tutorial-table-wrap">
          <table className="tutorial-data-table">
            <caption>Functional deck by player count</caption>
            <thead>
              <tr>
                <th scope="col">Players</th>
                <th scope="col">Lowest rank used</th>
                <th scope="col">Hidden cards</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>3</td><td>10</td><td>4</td></tr>
              <tr><td>4</td><td>9</td><td>2</td></tr>
              <tr><td>5</td><td>7</td><td>4</td></tr>
              <tr><td>6</td><td>6</td><td>2</td></tr>
              <tr><td>7</td><td>4</td><td>4</td></tr>
              <tr><td>8</td><td>3</td><td>2</td></tr>
            </tbody>
          </table>
        </div>
        <div className="tutorial-callout">
          <strong>Why hiding cards matter</strong>
          <p>
            Nobody plays hidden cards, but a hidden trump jack removes the jack
            scoring point, and hidden trump ranks determine which dealt trumps
            are the visible high and low.
          </p>
        </div>
      </>
    );
  }

  if (lessonId === "auction") {
    return (
      <>
        <p className="tutorial-lead">
          Before cards are played, each player gets exactly one chance to bid
          for the right to lead the round. A bid promises that the bidder or
          bidding team will score at least that many round points.
        </p>
        <ol className="tutorial-steps">
          <li>The dealer moves one seat clockwise each round.</li>
          <li>
            The player to the dealer&apos;s left bids first, continuing
            clockwise until the dealer acts last.
          </li>
          <li>
            A legal bid is an integer from <strong>1 through 6</strong> and
            must be higher than every earlier bid. A player may pass after a
            bid exists.
          </li>
          <li>
            Players may initially pass, but if everyone before the final
            bidder passed, that final bidder must bid.
          </li>
        </ol>
        <div className="tutorial-callout">
          <strong>A bid of 6</strong>
          <p>
            Six is the maximum, but it does not stop the auction early.
            Remaining players still take their turn; their only legal choice is
            to pass.
          </p>
        </div>
        <p>
          After the single bidding lap, the highest bidder wins the auction and
          leads the first trick.
        </p>
      </>
    );
  }

  if (lessonId === "trump") {
    return (
      <>
        <p className="tutorial-lead">
          Trump is not announced during bidding. The suit of the auction
          winner&apos;s first card becomes trump for the entire round. That
          opening card cannot be a joker because a joker has no suit.
        </p>
        <div className="tutorial-card-example">
          <div className="tutorial-card-example__header">
            <strong>Example trick</strong>
            <span>First lead made hearts trump</span>
          </div>
          <div className="tutorial-card-row">
            <TutorialCardPlay cardCode="10H" label="Lead" isTrump />
            <TutorialCardPlay cardCode="AS" label="Off-suit" />
            <TutorialCardPlay cardCode="J1" label="Joker" isTrump />
            <TutorialCardPlay cardCode="KH" label="Winner" isTrump />
          </div>
        </div>
        <h3>How a trick is won</h3>
        <ol className="tutorial-steps">
          <li>Highest card played in the round&apos;s trump suit wins.</li>
          <li>
            If no trump was played, the <strong>first joker played</strong>{" "}
            wins. A second joker cannot beat it.
          </li>
          <li>
            If there is neither trump nor a joker, the highest card in the
            suit led to this trick wins.
          </li>
        </ol>
        <p>
          After trump is set, any later trick leader may lead any card,
          including a joker. The winner of every trick leads the next one.
        </p>
      </>
    );
  }

  if (lessonId === "legal-plays") {
    return (
      <>
        <p className="tutorial-lead">
          During a trick, trump cards and jokers can always be played. Rules
          for ordinary cards depend on what was led.
        </p>
        <div className="tutorial-rule-grid tutorial-rule-grid--play">
          <article>
            <strong>Trump was led</strong>
            <span>
              Play trump or a joker if you hold either. You may discard a
              non-trump only when you have no trump-capable response.
            </span>
          </article>
          <article>
            <strong>Ordinary suit was led</strong>
            <span>
              You may trump or play a joker. If you play an ordinary card, you
              must follow the led suit when you have it.
            </span>
          </article>
          <article>
            <strong>You cannot follow</strong>
            <span>
              If an ordinary non-trump card led the trick and you hold none of
              its suit, any ordinary off-suit discard is legal.
            </span>
          </article>
          <article>
            <strong>A joker was led</strong>
            <span>Every card in your hand is legal.</span>
          </article>
        </div>
        <div className="tutorial-card-example">
          <div className="tutorial-card-example__header">
            <strong>Trump: hearts; lead: ace of diamonds</strong>
            <span>Legal cards are highlighted; king of spades is blocked.</span>
          </div>
          <div className="tutorial-card-row">
            <TutorialCardPlay cardCode="AD" label="Lead" />
            <TutorialCardPlay cardCode="9D" label="Follow suit" />
            <TutorialCardPlay cardCode="10H" label="Trump" isTrump />
            <TutorialCardPlay cardCode="J1" label="Joker" isTrump />
            <TutorialCardPlay cardCode="KS" label="Illegal discard" disabled />
          </div>
        </div>
        <div className="tutorial-callout">
          <strong>Trump-lead edge case</strong>
          <p>
            If diamonds are trump, diamonds are led, and your only
            trump-capable card is a joker, you must play that joker instead of
            discarding an ordinary off-suit card.
          </p>
        </div>
      </>
    );
  }

  if (lessonId === "scoring") {
    return (
      <>
        <p className="tutorial-lead">
          Up to six raw points are available after the sixth trick. Points go
          to a player or to the combined team when teams are enabled.
        </p>
        <div className="tutorial-points-grid">
          <article><strong>High</strong><span>Highest visible trump: 1 point.</span></article>
          <article><strong>Jack</strong><span>Jack of trump: 1 point, unless hidden.</span></article>
          <article><strong>Low</strong><span>Lowest visible trump: 1 point for its original holder.</span></article>
          <article><strong>Jokers</strong><span>Each captured joker: 1 point, up to 2.</span></article>
          <article><strong>Game</strong><span>Unique highest game-card total: 1 point.</span></article>
        </div>
        <div className="tutorial-callout">
          <strong>Low is different</strong>
          <p>
            High, jack, jokers, and game depend on possession or captured
            cards. Low belongs to whoever was originally dealt the lowest
            visible trump and played it, even when another player captures its
            trick.
          </p>
        </div>
        <h3>Counting game</h3>
        <div className="tutorial-value-row" aria-label="Game card values">
          <span><strong>10</strong> = 10</span>
          <span><strong>J</strong> = 1</span>
          <span><strong>Q</strong> = 2</span>
          <span><strong>K</strong> = 3</span>
          <span><strong>A</strong> = 4</span>
          <span><strong>2-9</strong> = 0</span>
        </div>
        <p>
          Only a unique highest captured-card total earns game. If the leading
          total is tied, no one receives that point. Visible trump means a
          trump card in the functional deck that was not hidden in the deal.
        </p>
      </>
    );
  }

  if (lessonId === "match") {
    return (
      <>
        <p className="tutorial-lead">
          Raw points are applied differently to the bidding unit and everyone
          else. The standard target score is 21.
        </p>
        <div className="tutorial-rule-grid">
          <article>
            <strong>Bid made</strong>
            <span>
              If the bidder scores at least its bid, it adds its full raw round
              total, not just the bid.
            </span>
          </article>
          <article>
            <strong>Bid missed</strong>
            <span>
              If the bidder falls short, it subtracts the bid amount from its
              match score.
            </span>
          </article>
          <article>
            <strong>Non-bidders</strong>
            <span>
              Every other scoring unit adds its raw points, but cannot rise
              above 20 in a game to 21.
            </span>
          </article>
          <article>
            <strong>Winning</strong>
            <span>
              Only the unit that won the auction can reach the target and win
              at the end of that round.
            </span>
          </article>
        </div>
        <div className="tutorial-score-example">
          <strong>Contract examples</strong>
          <p>
            Bid 3 and score 5: add 5. Bid 4 and score 3: subtract 4. A
            non-bidding team at 19 that scores 3 finishes the round at 20, not
            22.
          </p>
        </div>
      </>
    );
  }

  return (
    <>
      <p className="tutorial-lead">
        You can now follow a round without guessing: cards are dealt, players
        bid once, the bidder&apos;s opening lead sets trump, six tricks are
        played legally, points are counted, and the bidder&apos;s contract
        decides match movement.
      </p>
      <div className="tutorial-recap">
        <h3>At the table, remember</h3>
        <ol className="tutorial-steps">
          <li>
            Bid for points you believe your hand or team can secure, remembering
            that losing a bid costs its full value.
          </li>
          <li>
            If you win the auction, choose your first lead carefully because
            its suit is trump for all six tricks.
          </li>
          <li>
            During tricks, track trump and jokers first, then led suit; do not
            make an off-suit discard when the follow rules prohibit it.
          </li>
          <li>
            Track high, jack, low, both jokers, and captured game values while
            aiming to make the bid and eventually reach 21.
          </li>
        </ol>
      </div>
      <p>
        Practice offers generated decisions with bot feedback. Play starts a
        table against bots or with other people.
      </p>
    </>
  );
}

type TutorialQuestionProps = {
  lesson: TutorialLesson;
  selectedAnswer: number | undefined;
  onSelectAnswer: (answerIndex: number) => void;
};

function TutorialQuestion({
  lesson,
  selectedAnswer,
  onSelectAnswer,
}: TutorialQuestionProps) {
  const answer =
    selectedAnswer === undefined ? null : lesson.answers[selectedAnswer];

  return (
    <section className="tutorial-question">
      <span className="eyebrow">Check your understanding</span>
      <h3>{lesson.question}</h3>
      <div className="tutorial-answer-grid">
        {lesson.answers.map((option, index) => (
          <button
            key={option.label}
            type="button"
            className={[
              "tutorial-answer",
              selectedAnswer === index ? "is-selected" : "",
              selectedAnswer === index && option.isCorrect ? "is-correct" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            aria-pressed={selectedAnswer === index}
            onClick={() => onSelectAnswer(index)}
          >
            {option.label}
          </button>
        ))}
      </div>
      {answer ? (
        <p
          className={[
            "tutorial-feedback",
            answer.isCorrect ? "tutorial-feedback--correct" : "",
          ]
            .filter(Boolean)
            .join(" ")}
          role="status"
        >
          <strong>{answer.isCorrect ? "Correct." : "Try again."}</strong>{" "}
          {answer.feedback}
        </p>
      ) : (
        <p className="tutorial-feedback">
          Choose an answer to unlock the next lesson.
        </p>
      )}
    </section>
  );
}

function TutorialPage({
  onNavigateHome,
  onNavigatePlay,
  onNavigateLearn,
  onNavigateDonate,
}: PageNavigationProps) {
  const [lessonIndex, setLessonIndex] = useState(0);
  const [unlockedIndex, setUnlockedIndex] = useState(0);
  const [answers, setAnswers] = useState<
    Partial<Record<TutorialLessonId, number>>
  >({});
  const [hasCompletedTutorial, setHasCompletedTutorial] = useState(false);
  const lesson = TUTORIAL_LESSONS[lessonIndex];
  const selectedAnswer = answers[lesson.id];
  const answerIsCorrect =
    selectedAnswer !== undefined &&
    lesson.answers[selectedAnswer]?.isCorrect === true;
  const isLastLesson = lessonIndex === TUTORIAL_LESSONS.length - 1;
  const progressPercent = Math.round(
    ((lessonIndex + (answerIsCorrect ? 1 : 0)) / TUTORIAL_LESSONS.length) *
      100,
  );

  function handleContinue() {
    if (!answerIsCorrect) {
      return;
    }
    if (isLastLesson) {
      setHasCompletedTutorial(true);
      return;
    }

    const nextIndex = lessonIndex + 1;
    setUnlockedIndex((current) => Math.max(current, nextIndex));
    setLessonIndex(nextIndex);
  }

  return (
    <main className="public-shell tutorial-shell">
      <section className="hero-card">
        <div className="hero-card__branding">
          <BrandLogo className="brand-logo--badge" />
          <div>
            <span className="eyebrow">Learn / Tutorial</span>
            <h1>How to play Smear</h1>
          </div>
        </div>
        <div className="top-actions">
          <button type="button" className="ghost-button" onClick={onNavigateLearn}>
            Learn
          </button>
          <button type="button" className="ghost-button" onClick={onNavigateHome}>
            Home
          </button>
          <button type="button" className="ghost-button" onClick={onNavigatePlay}>
            Play
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigateDonate}
          >
            Donate
          </button>
        </div>
      </section>

      <section className="tutorial-layout">
        <aside className="tutorial-sidebar" aria-label="Tutorial lessons">
          <div>
            <span className="eyebrow">Progress</span>
            <strong>{progressPercent}% complete</strong>
          </div>
          <div className="tutorial-progress-track" aria-hidden="true">
            <span style={{ width: `${progressPercent}%` }} />
          </div>
          <nav className="tutorial-step-list">
            {TUTORIAL_LESSONS.map((item, index) => {
              const selected = index === lessonIndex;
              const lessonAnswer = answers[item.id];
              const completed =
                lessonAnswer !== undefined &&
                item.answers[lessonAnswer]?.isCorrect === true;

              return (
                <button
                  key={item.id}
                  type="button"
                  className={[
                    "tutorial-step-button",
                    selected ? "is-selected" : "",
                    completed ? "is-complete" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  onClick={() => setLessonIndex(index)}
                  disabled={index > unlockedIndex}
                  aria-current={selected ? "step" : undefined}
                >
                  <span>{item.eyebrow}</span>
                  <strong>{item.title}</strong>
                </button>
              );
            })}
          </nav>
        </aside>

        <article className="tutorial-content">
          <header className="tutorial-content__header">
            <span className="eyebrow">{lesson.eyebrow}</span>
            <h2>{lesson.title}</h2>
            <p>{lesson.overview}</p>
          </header>

          <TutorialLessonContent lessonId={lesson.id} />

          <TutorialQuestion
            lesson={lesson}
            selectedAnswer={selectedAnswer}
            onSelectAnswer={(answerIndex) =>
              setAnswers((current) => ({
                ...current,
                [lesson.id]: answerIndex,
              }))
            }
          />

          {hasCompletedTutorial && isLastLesson ? (
            <section className="tutorial-complete" aria-live="polite">
              <span className="eyebrow">Tutorial complete</span>
              <h3>You are ready to play a round.</h3>
              <p>
                Use Practice to check individual decisions against a bot, or
                open a table and put the full round together.
              </p>
              <div className="tutorial-complete__actions">
                <button
                  type="button"
                  className="cta-button"
                  onClick={() => setHashView("learn/practice")}
                >
                  Practice positions
                </button>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={onNavigatePlay}
                >
                  Play Smear
                </button>
              </div>
            </section>
          ) : null}

          <div className="tutorial-controls">
            <button
              type="button"
              className="ghost-button"
              onClick={() => setLessonIndex((current) => current - 1)}
              disabled={lessonIndex === 0}
            >
              Previous lesson
            </button>
            <button
              type="button"
              className="cta-button"
              onClick={handleContinue}
              disabled={!answerIsCorrect || (isLastLesson && hasCompletedTutorial)}
            >
              {isLastLesson ? "Finish tutorial" : "Next lesson"}
            </button>
          </div>
        </article>
      </section>

      <SiteFooter />
    </main>
  );
}

function PracticePage({
  onNavigateHome,
  onNavigatePlay,
  onNavigateLearn,
  onNavigateDonate,
}: PageNavigationProps) {
  const client = useMemo(
    () => createApiClient({ apiBaseUrl: API_BASE_URL, sessionId: null }),
    [],
  );
  const [challenge, setChallenge] = useState<LearnChallenge | null>(null);
  const [selectedAction, setSelectedAction] = useState<LearnAction | null>(
    null,
  );
  const [availableLearnBots, setAvailableLearnBots] = useState<ReadyBot[]>([]);
  const [selectedLearnBotId, setSelectedLearnBotId] =
    useState(DEFAULT_LEARN_BOT_ID);
  const [isLoadingChallenge, setIsLoadingChallenge] = useState(true);
  const [learnError, setLearnError] = useState<string | null>(null);
  const [botLoadError, setBotLoadError] = useState<string | null>(null);

  const loadChallenge = useCallback(async () => {
    setIsLoadingChallenge(true);
    setLearnError(null);
    setSelectedAction(null);
    try {
      const nextChallenge = await client.fetchLearnChallenge(
        undefined,
        selectedLearnBotId,
      );
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
  }, [client, selectedLearnBotId]);

  useEffect(() => {
    let isActive = true;

    async function loadBots() {
      setBotLoadError(null);
      try {
        const response = await client.fetchBots();
        if (!isActive) {
          return;
        }
        setAvailableLearnBots(response.bots);
        if (response.bots.length > 0) {
          setSelectedLearnBotId((currentBotId) =>
            response.bots.some((bot) => bot.id === currentBotId)
              ? currentBotId
              : response.bots[0].id,
          );
        }
      } catch (error) {
        if (!isActive) {
          return;
        }
        setBotLoadError(
          error instanceof Error ? error.message : "Could not load bots.",
        );
      }
    }

    void loadBots();

    return () => {
      isActive = false;
    };
  }, [client]);

  useEffect(() => {
    void loadChallenge();
  }, [loadChallenge]);

  const actor = challenge
    ? (challenge.state.round.players.find(
        (player) => player.name === challenge.actor_name,
      ) ?? null)
    : null;
  const selectedKey = selectedAction ? getActionKey(selectedAction) : null;
  const bestKey = challenge ? getActionKey(challenge.best_action) : null;
  const selectedMatchesBest =
    selectedKey !== null && bestKey !== null && selectedKey === bestKey;
  const currentTrick = challenge?.state.round.current_trick ?? null;
  const learnBotOptions =
    availableLearnBots.length > 0
      ? availableLearnBots
      : [
          {
            id: DEFAULT_LEARN_BOT_ID,
            label: challenge?.best_bot_label ?? "Optimal Bot",
            description: "Adaptive hidden-information minimax.",
          },
        ];
  const selectedLearnBot =
    learnBotOptions.find((bot) => bot.id === selectedLearnBotId) ?? null;

  return (
    <main className="public-shell learn-shell">
      <section className="hero-card">
        <div className="hero-card__branding">
          <BrandLogo className="brand-logo--badge" />
          <div>
            <span className="eyebrow">Learn / Practice</span>
            <h1>Practice a position</h1>
          </div>
        </div>
        <div className="top-actions">
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigateLearn}
          >
            Learn
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigateHome}
          >
            Home
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigatePlay}
          >
            Play
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onNavigateDonate}
          >
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
                  : (challenge?.prompt ?? "No position loaded")}
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
                  <strong>
                    {challenge.state.auction.current_high_bid ?? "-"}
                  </strong>
                </div>
                <div className="status-pill">
                  <span>High bidder</span>
                  <strong>
                    {challenge.state.auction.highest_bidder_name ?? "-"}
                  </strong>
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
                    <div className="empty-inline">
                      No cards have been played to this trick.
                    </div>
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
                          card.is_joker ||
                          card.suit === challenge.state.round.trump
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
              <h2>
                {isLoadingChallenge
                  ? "Finding a position."
                  : "Try loading a new position."}
              </h2>
            </div>
          )}
        </div>

        <aside className="learn-actions">
          <div>
            <span className="eyebrow">Choose</span>
            <h2>Your options</h2>
          </div>

          <label className="field learn-bot-field">
            <span>Learning from</span>
            <select
              value={selectedLearnBotId}
              onChange={(event) => {
                setSelectedLearnBotId(event.target.value);
              }}
              disabled={isLoadingChallenge && availableLearnBots.length === 0}
            >
              {learnBotOptions.map((bot) => (
                <option key={bot.id} value={bot.id}>
                  {bot.label}
                </option>
              ))}
            </select>
          </label>

          {selectedLearnBot ? (
            <p className="learn-bot-detail">{selectedLearnBot.description}</p>
          ) : null}

          {botLoadError ? (
            <div className="empty-inline">{botLoadError}</div>
          ) : null}

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
            <div className="empty-inline">
              Options will appear after the position loads.
            </div>
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
              <h2>
                {selectedMatchesBest ? "Same choice" : "Different choice"}
              </h2>
              <p>
                You chose <strong>{selectedAction.label}</strong>.{" "}
                {challenge.best_bot_label} would choose{" "}
                <strong>{challenge.best_action.label}</strong>.
              </p>
              <p className="learn-result__explanation">
                {challenge.best_action_explanation}
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

      <SiteFooter />
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
    return <PlayPage {...navigation} />;
  }

  if (view === "learn") {
    return (
      <LearnPage
        {...navigation}
        onChoosePractice={() => setHashView("learn/practice")}
        onChooseTutorial={() => setHashView("learn/tutorial")}
      />
    );
  }

  if (view === "learn/practice") {
    return <PracticePage {...navigation} />;
  }

  if (view === "learn/tutorial") {
    return <TutorialPage {...navigation} />;
  }

  if (view === "donate") {
    return <DonationPage {...navigation} />;
  }

  return <LandingPage {...navigation} />;
}
