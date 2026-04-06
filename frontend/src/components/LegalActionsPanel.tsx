import type { LegalAction } from "../types";

type LegalActionsPanelProps = {
  actions: LegalAction[];
  disabled?: boolean;
  onAction: (action: LegalAction) => void;
};

function actionLabel(action: LegalAction) {
  if (action.type === "bid") {
    return `Bid ${action.value}`;
  }

  if (action.type === "play_card") {
    return `Play ${action.card_id}`;
  }

  if (action.type === "choose_trump") {
    return `Choose ${action.suit}`;
  }

  return action.type;
}

export function LegalActionsPanel({ actions, disabled = false, onAction }: LegalActionsPanelProps) {
  return (
    <section className="panel">
      <header className="panel-header">
        <div>
          <h2>Legal Actions</h2>
          <p>Buttons call the backend and then reload the state.</p>
        </div>
      </header>

      <div className="action-grid">
        {actions.length > 0 ? (
          actions.map((action, index) => (
            <button
              key={`${action.type}-${action.value ?? "none"}-${action.card_id ?? "none"}-${action.suit ?? "none"}-${index}`}
              className="action-button"
              disabled={disabled}
              onClick={() => onAction(action)}
              type="button"
            >
              {actionLabel(action)}
            </button>
          ))
        ) : (
          <p className="muted">No legal actions are available for this state.</p>
        )}
      </div>
    </section>
  );
}
