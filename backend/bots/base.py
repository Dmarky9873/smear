from __future__ import annotations
from threading import Lock

try:
    from backend.models import AuctionEvent, AuctionState, Card, Player, RoundState
except ImportError:
    from models import AuctionEvent, AuctionState, Card, Player, RoundState


class BotPlayer(Player):
    """Abstract base bot class"""

    def __init__(self, name: str, cards: set[Card] = None):
        super().__init__(name, cards)
        self._progress_lock = Lock()
        self._progress_active = False
        self._progress_label: str | None = None
        self._progress_detail: str | None = None
        self._progress_completed_units = 0
        self._progress_total_units = 0

    def choose_card(self, round_state: RoundState) -> Card:
        raise NotImplementedError

    def choose_auction_action(self, auction_state: AuctionState) -> AuctionEvent:
        raise NotImplementedError

    def begin_progress(
        self,
        *,
        label: str,
        total_units: int,
        detail: str | None = None,
    ) -> None:
        with self._progress_lock:
            self._progress_active = True
            self._progress_label = label
            self._progress_detail = detail
            self._progress_completed_units = 0
            self._progress_total_units = max(total_units, 0)

    def update_progress(
        self,
        *,
        completed_units: int,
        detail: str | None = None,
    ) -> None:
        with self._progress_lock:
            if not self._progress_active:
                return
            self._progress_completed_units = max(completed_units, 0)
            if detail is not None:
                self._progress_detail = detail

    def clear_progress(self) -> None:
        with self._progress_lock:
            self._progress_active = False
            self._progress_label = None
            self._progress_detail = None
            self._progress_completed_units = 0
            self._progress_total_units = 0

    def get_progress_snapshot(self) -> dict | None:
        with self._progress_lock:
            if not self._progress_active:
                return None

            total_units = self._progress_total_units
            completed_units = min(self._progress_completed_units, total_units)
            percent_complete = (
                (completed_units / total_units) * 100 if total_units > 0 else 0.0
            )
            return {
                "label": self._progress_label,
                "detail": self._progress_detail,
                "completed_units": completed_units,
                "total_units": total_units,
                "percent_complete": percent_complete,
            }
