from models import Player, Team, Card, Deck, RoundState
from constants import RANKS, HAND_SIZE


class Game:

    hand_size: int = HAND_SIZE
    players: dict[str, Player]
    player_order: list[str] = []
    teams: set[Team] = None
    deck: Deck
    trump_suit: str = None

    def __init__(self, num_players: int, player_names: list[str]):
        if num_players > 8 or num_players < 3:
            raise ValueError(
                f"the number of players must be in [3, 8], you gave {num_players}")

        self._num_players = num_players
        self.players = {}
        self._low, self._num_hiding, self._num_dealt = self._calculate_low()
        self.deck = Deck(self._low)

        if len(player_names) != self.num_players:
            raise ValueError(
                f"you did not provide an adaquate number of names")

        self.player_order = player_names.copy()

    def _calculate_low(self) -> tuple[str, int, int]:
        dealt = Game.hand_size * self.num_players
        best_low = None
        best_hiding = None
        best_diff = float("inf")

        for i, rank in enumerate(RANKS):
            remaining_ranks = RANKS[i:]
            deck_size = 4 * len(remaining_ranks) + 2
            hiding = deck_size - dealt

            if hiding <= 0:
                continue

            diff = abs(hiding - 2)

            if diff < best_diff:
                best_diff = diff
                best_low = rank
                best_hiding = hiding

        return best_low, best_hiding, dealt

    @property
    def num_players(self):
        return self._num_players

    @property
    def low(self):
        return self._low

    @property
    def num_hiding(self):
        return self._num_hiding

    @property
    def num_dealt(self):
        return self._num_dealt

    def view_state(self):
        if self.players == {}:
            print("no game has been initialized. Call `new_game`.")
        else:
            print("Players:")
            for name, player in self.players.items():
                print(
                    f"{name}: cards: {player.cards}, captured cards: {player.captured_cards}")

            print(f"Trump suit: {self.trump_suit}")

    def deal_cards(self):
        self.deck.shuffle()
        deck_copy = self.deck.deck.copy()

        for player in self.player_order:
            self.players[player].receive_new_hand(
                {deck_copy.pop() for _ in range(Game.hand_size)})

    def init_new_game(self):

        print(f"creating a game with a low of {self.low}...")

        for player in self.player_order:
            self.players[player] = Player(
                player, set())

        self.deal_cards()

        print(f"initialized game with {len(self.player_order)} players.")
        for player in self.player_order:
            print(self.players[player])


class Simulator:
    game: Game

    def __init__(self, game: Game):
        self._game = game


def get_legal_actions(state: RoundState) -> set[Card]:
    hand = state.current_player.cards

    if not state.current_trick.plays:
        return set(hand)

    lead_card = state.current_trick.plays[0][1]

    # If trump was led, must play trump if possible.
    if state.trump is not None and not lead_card.is_joker and lead_card.suit == state.trump:
        trump_cards = {
            card for card in hand
            if not card.is_joker and card.suit == state.trump
        }
        if trump_cards:
            return trump_cards
        return set(hand)

    # If a non-trump, non-joker suit was led, must follow that suit if possible.
    if not lead_card.is_joker and lead_card.suit is not None:
        same_suit_cards = {
            card for card in hand
            if not card.is_joker and card.suit == lead_card.suit
        }
        if same_suit_cards:
            return same_suit_cards
        return set(hand)

    # If a joker was led, anything may be played.
    return set(hand)
