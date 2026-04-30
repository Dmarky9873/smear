import { PlayingCard } from "./PlayingCard";
import type { TrickState } from "../types";

type CurrentTrickPanelProps = {
  trick: TrickState;
};

export function CurrentTrickPanel({ trick }: CurrentTrickPanelProps) {
  return (
    <section className="panel">
      <h2>Current Trick</h2>
      <p>
        Leader: <strong>{trick.leader_name}</strong>
      </p>
      <div className="list-block">
        {trick.plays.length === 0 ? (
          <span className="muted">No cards have been played yet.</span>
        ) : (
          trick.plays.map((play, index) => (
            <div key={`${play.player_name}-${play.card.code}-${index}`} className="list-row">
              <span>{play.player_name}</span>
              <PlayingCard card={play.card} compact />
            </div>
          ))
        )}
      </div>
    </section>
  );
}
