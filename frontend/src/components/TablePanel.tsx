import { PlayingCard } from "./PlayingCard";
import type { GameState } from "../types";

type TablePanelProps = {
  state: GameState;
};

export function TablePanel({ state }: TablePanelProps) {
  return (
    <section className="panel">
      <header className="panel-header">
        <div>
          <h2>Table</h2>
          <p>Current trick and completed trick summary.</p>
        </div>
        <div className="stacked-stats">
          <span>Completed tricks: {state.completed_tricks.length}</span>
          <span>Lead player: {state.leading_player_id ?? "—"}</span>
        </div>
      </header>

      <div className="trick-grid">
        {state.current_trick.length > 0 ? (
          state.current_trick.map((playedCard) => (
            <div key={`${playedCard.player_id}-${playedCard.card.id}`} className="trick-slot">
              <span className="muted">Player {playedCard.player_id}</span>
              <PlayingCard card={playedCard.card} trumpSuit={state.trump_suit} />
            </div>
          ))
        ) : (
          <p className="muted">No cards on the table yet.</p>
        )}
      </div>
    </section>
  );
}
