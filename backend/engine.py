from models import Player, Team, Card, Deck, RoundState, TrickState, Play
from constants import RANKS, HAND_SIZE


class Game:

    hand_size: int = HAND_SIZE
    teams: set[Team] = None

    def __init__(self, num_players: int, player_names: list[str], teams: set[tuple]):

        if num_players > 8 or num_players < 3:
            raise ValueError(
                f"the number of players must be in [3, 8], you gave {num_players}")

        self._low, self._num_hiding, self._num_dealt = self._calculate_low(
            num_players)

        deck = Deck(self._low)

        print(f"creating a game with a low of {self._low}...")

        if len(player_names) != num_players:
            raise ValueError(
                f"you did not provide an adaquate number of names")

        players = [Player(name, set()) for name in player_names]
        player_dict = dict()
        for player in players:
            player_dict[player.name] = player

        curr_player = players[0]

        hiding_cards = set(self.deal_cards(deck, players))

        teams_to_add = []
        for team in teams:
            new_team = Team([player_dict[name] for name in team], set())
            teams_to_add.append(new_team)

        first_trick = TrickState(players[0], [], players)

        self._round_state = RoundState(players,
                                       curr_player, 0, None, first_trick, hiding_cards, [], teams_to_add, deck)

        print(f"initialized game with {num_players} players.")
        for player in players:
            print(player)

    def _calculate_low(self, num_players) -> tuple[str, int, int]:
        dealt = Game.hand_size * num_players
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
    def low(self):
        return self._low

    def view_state(self):
        if self._round_state.players == []:
            print("no game has been initialized. Call `new_game`.")
        else:
            print("Players:")
            for player in self._round_state.players:
                print(
                    f"{player.name}: cards: {player.cards}, captured cards: {player.captured_cards}")

            print(f"Trump suit: {self._round_state.trump}")

    def get_state(self) -> RoundState:
        if self._round_state.current_player == None:
            raise ValueError(f"initialize a game before getting the state")
        return self._round_state

    def apply_trick_action(self, action: Play):

        if action.player != self._round_state.current_player:
            raise ValueError(
                f"action with player {action.player} was attempted despite {self._round_state.current_player} being the current player")

        if action.card not in get_legal_actions(self.get_state):
            raise ValueError(
                f"action with card {action.card} attepted despite card not being in legal moves")

        curr_trick = self._round_state.current_trick
        self._round_state.current_player.play_card(action.card)
        curr_trick.plays.append(action)

        if len(self._round_state.current_trick.plays) == len(self._round_state.players):

            trick_winner = get_trick_winner(curr_trick)

            for play in curr_trick.plays:
                trick_winner.capture(play.card)

            self._round_state.current_trick = TrickState(
                trick_winner, [], curr_trick.players)
            self._round_state.trick_history.append(
                curr_trick)

    def deal_cards(self, deck: Deck, players: set[Player] | list[Player]) -> list[Card]:
        """Shuffles and deals cards to the players within the game

        Returns:
            list[Card]: Cards that are hiding based on this shuffle
        """
        deck.shuffle()
        deck_copy = deck.get_copy()

        for player in players:
            player.receive_new_hand(
                {deck_copy.pop() for _ in range(Game.hand_size)})

        return deck_copy


class Simulator:
    game: Game

    def __init__(self, game: Game):
        self._game = game


def get_legal_actions(state: RoundState) -> set[Card]:
    hand = state.current_player.cards

    if not state.current_trick.plays:
        return set(hand)

    lead_card = state.current_trick.plays[0].card

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


def get_trick_winner(state: TrickState) -> Player:
    if not state.is_terminal:
        raise ValueError(f"trick is not in terminal state")

    ...
