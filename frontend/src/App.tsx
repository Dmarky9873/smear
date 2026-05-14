import { useEffect, useState } from "react";

import {
  loadOrCreateSessionId,
  useSmearGame,
} from "@smear/web-core";
import { DebugModeView } from "./components/DebugModeView";
import { PlayModeView } from "./components/PlayModeView";

type AppMode = "play" | "debug";
type ThemeMode = "light" | "dark";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export default function App() {
  const [mode, setMode] = useState<AppMode>("play");
  const [sessionId] = useState(() =>
    loadOrCreateSessionId("smear-debug-session-id"),
  );
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") {
      return "light";
    }

    return window.localStorage.getItem("smear-theme") === "dark"
      ? "dark"
      : "light";
  });

  const {
    availableBots,
    awaitingNextTrick,
    bidActions,
    botActionDelayMs,
    botProgress,
    botThinkingName,
    canPassAuction,
    currentPlayer,
    currentTurnName,
    error,
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
    isLoading,
    numPlayers,
    playActions,
    playerBots,
    playerNames,
    revealedTrick,
    score,
    setBotActionDelayMs,
    setNumPlayers,
    setTeamsInput,
    state,
    teamsInput,
    turnPlayer,
  } = useSmearGame({
    apiBaseUrl: API_BASE_URL,
    botAutomationEnabled: mode === "play",
    sessionId,
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("smear-theme", theme);
  }, [theme]);

  function confirmNewGameIfNeeded(): boolean {
    if (!state) {
      return true;
    }

    return window.confirm("Are you sure you want to start a new game?");
  }

  async function handleConfirmedNewGameDebug() {
    if (!confirmNewGameIfNeeded()) {
      return;
    }

    await handleNewGameDebug();
  }

  async function handleConfirmedNewGamePlay() {
    if (!confirmNewGameIfNeeded()) {
      return;
    }

    await handleNewGamePlay();
  }

  return (
    <>
      <header className="mode-switcher">
        <div className="mode-switcher__copy">
          <strong>Frontend mode</strong>
          <span>Play is table-first. Debug keeps the full state inspector.</span>
        </div>
        <div className="mode-switcher__actions">
          <button
            type="button"
            className="mode-switcher__theme-toggle"
            aria-pressed={theme === "dark"}
            onClick={() =>
              setTheme((current) => (current === "dark" ? "light" : "dark"))
            }
          >
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
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
        </div>
      </header>

      {mode === "debug" ? (
        <DebugModeView
          numPlayers={numPlayers}
          playerNames={playerNames}
          playerBots={playerBots}
          teamsInput={teamsInput}
          availableBots={availableBots}
          botProgress={botProgress}
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
            void handleConfirmedNewGameDebug();
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
          botProgress={botProgress}
          state={state}
          score={score}
          error={error}
          isBusy={isLoading}
          botThinkingName={botThinkingName}
          revealedTrick={revealedTrick}
          awaitingNextTrick={awaitingNextTrick}
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
          onAdvanceToNextTrick={handleAdvanceToNextTrick}
          onNewGame={() => {
            void handleConfirmedNewGamePlay();
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
