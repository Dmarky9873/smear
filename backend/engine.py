from backend.models import Player, Team, Card, Deck, RANKS


class Game:

    hand_size: int = 6
    players: dict[str: Player]
    player_order: list[str]
    teams: set[Team]
    deck: Deck
    trump_suit: str = None

    def __init__(self, num_players: int):
        if num_players > 8 or num_players < 3:
            raise ValueError(
                f"the number of players must be in [3, 8], you gave {num_players}")

        self._num_players = num_players
        self._low, self._num_hiding, self._num_dealt = self._calculate_low()
        self.deck = Deck(self._low)

    def _calculate_low(self) -> tuple[int, int, int]:
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
        return self.num_dealt

    def view_state(self):
        if self.players == {}:
            print("no game has been initialized. Call `new_game`.")
        else:
            print("Players:")
            for name, player in self.players.items():
                print(
                    f"{name}: cards: {player.cards}, captured cards: {player.captured_cards}")

            print(f"Trump suit: {self.trump_suit}")

    def new_game(self, player_names: list[str]):
        if len(player_names) != self.num_players:
            raise ValueError(
                f"you did not provide an adaquate number of names")

        print(f"creating a game with a low of {self.low}...")

        self.deck.shuffle()

        deck_copy = self.deck.deck.copy()
        for player in player_names:
            self.player_order.append(player)
            self.players[player] = Player(
                player, {deck_copy.pop() for _ in range(Game.hand_size)})
