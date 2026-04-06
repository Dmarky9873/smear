import type { GameState } from "../types";

type TopStatusBarProps = {
  state: GameState;
};

function joinScoreMap(values: Record<string, number>) {
  const entries = Object.entries(values);
  if (entries.length === 0) {
    return "—";
  }

  return entries.map(([key, value]) => `${key}: ${value}`).join(" · ");
}

export function TopStatusBar({ state }: TopStatusBarProps) {
  const currentPlayer = state.players.find((player) => player.id === state.current_player_id);
  const winningBidder = state.players.find((player) => player.id === state.winning_bidder_id);
  const dealer = state.players.find((player) => player.id === state.dealer_id);

  return (
    <section className="panel status-bar">
      <header className="panel-header">
        <div>
          <h1>Smear Debug Harness</h1>
          <p>Local placeholder state viewer for backend integration work.</p>
        </div>
      </header>

      <div className="status-grid">
        <div>
          <span className="label">Game ID</span>
          <strong>{state.game_id}</strong>
        </div>
        <div>
          <span className="label">Phase</span>
          <strong>{state.phase}</strong>
        </div>
        <div>
          <span className="label">Current Player</span>
          <strong>{currentPlayer?.name ?? `Player ${state.current_player_id}`}</strong>
        </div>
        <div>
          <span className="label">Dealer</span>
          <strong>{dealer?.name ?? "—"}</strong>
        </div>
        <div>
          <span className="label">Current Bid</span>
          <strong>{state.current_bid ?? "—"}</strong>
        </div>
        <div>
          <span className="label">Winning Bidder</span>
          <strong>{winningBidder?.name ?? "—"}</strong>
        </div>
        <div>
          <span className="label">Trump Suit</span>
          <strong>{state.trump_suit ?? "—"}</strong>
        </div>
        <div>
          <span className="label">Scores</span>
          <strong>{joinScoreMap(state.scores)}</strong>
        </div>
        <div>
          <span className="label">Round Points</span>
          <strong>{joinScoreMap(state.round_points)}</strong>
        </div>
      </div>
    </section>
  );
}
