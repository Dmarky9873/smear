import { PlayingCard } from "./PlayingCard";
import type { Player } from "../types";

type PlayerPanelProps = {
  player: Player;
  isCurrent: boolean;
  trumpSuit: string | null;
};

export function PlayerPanel({ player, isCurrent, trumpSuit }: PlayerPanelProps) {
  return (
    <article className={`panel player-panel ${isCurrent ? "panel-current" : ""}`}>
      <header className="panel-header">
        <div>
          <h3>{player.name}</h3>
          <p>
            Player {player.id}
            {player.team_id !== null ? ` · Team ${player.team_id}` : ""}
          </p>
        </div>
        <span className={`turn-badge ${isCurrent ? "turn-badge-active" : ""}`}>
          {isCurrent ? "Current Turn" : "Waiting"}
        </span>
      </header>

      <div className="meta-grid">
        <span>Bid: {player.bid ?? "—"}</span>
        <span>Passed: {player.has_passed ? "Yes" : "No"}</span>
        <span>Captured: {player.captured_cards.length}</span>
      </div>

      <div className="card-row">
        {player.hand.length > 0 ? (
          player.hand.map((card) => (
            <PlayingCard key={card.id} card={card} trumpSuit={trumpSuit} />
          ))
        ) : (
          <p className="muted">No cards left in hand.</p>
        )}
      </div>
    </article>
  );
}
