from models import Player, Team, Card, Deck, RoundState, TrickState, Play, get_max_card
from constants import RANKS, HAND_SIZE


class Game:

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

        first_trick = TrickState(players[0], [], players, None)

        self._round_state = RoundState(players,
                                       curr_player, None, first_trick, hiding_cards, [], teams_to_add, deck)

        print(f"initialized game with {num_players} players.")
        for player in players:
            print(player)

    def _calculate_low(self, num_players) -> tuple[str, int, int]:
        dealt = HAND_SIZE * num_players
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
        print("Players:")
        for player in self._round_state.players:
            print(
                f"{player.name}: cards: {player.cards}, captured cards: {player.captured_cards}")

        print(f"Trump suit: {self._round_state.trump}")

    def apply_trick_action(self, action: Play) -> bool:
        """ Applies action and returns whether action resulted in a terminal trick state.

        Args:
            action (Play): The action to apply

        Raises:
            ValueError: If the action attempted was not with the current player, raise ValueError
            ValueError: If the action attempted was not with a legal card, raise ValueError

        Returns:
            bool: Whether the action resulted in a terminal trick state.
        """

        if action.player != self._round_state.current_player:
            raise ValueError(
                f"action with player {action.player} was attempted despite {self._round_state.current_player} being the current player")

        if action.card not in get_legal_actions(self._round_state):
            raise ValueError(
                f"action with card {action.card} attepted despite card not being in legal moves")

        curr_player = self._round_state.current_player
        curr_trick = self._round_state.current_trick
        curr_player.play_card(action.card)
        curr_trick.plays.append(action)

        players = self._round_state.players

        if self._round_state.current_trick.is_terminal:

            trick_winner = get_trick_winner(curr_trick)

            self._round_state.current_player = trick_winner

            for play in curr_trick.plays:
                trick_winner.capture(play.card)

            self._round_state.current_trick = TrickState(
                trick_winner, [], curr_trick.players, self._round_state.trump)
            self._round_state.trick_history.append(
                curr_trick)

            return True

        else:
            self._round_state.current_player = players[(
                players.index(curr_player) + 1) % len(players)]

            return False

    @property
    def curr_player(self):
        return self._round_state.current_player

    def deal_cards(self, deck: Deck, players: set[Player] | list[Player]) -> list[Card]:
        """Shuffles and deals cards to the players within the game

        Returns:
            list[Card]: Cards that are hiding based on this shuffle
        """
        deck.shuffle()
        deck_copy = deck.get_copy()

        for player in players:
            player.receive_new_hand(
                {deck_copy.pop() for _ in range(HAND_SIZE)})

        return deck_copy


def get_legal_actions(state: RoundState) -> set[Card]:
    hand = state.current_player.cards

    if not state.current_trick.plays:
        return set(hand)

    # Check if any trump has been played (includes trumping in)
    trump_played = any(
        not play.card.is_joker and play.card.suit == state.trump
        for play in state.current_trick.plays
    )

    # Check if any joker has been played
    joker_played = any(
        play.card.is_joker for play in state.current_trick.plays)

    # If trump or joker was played (including when someone trumps in), must play trump or joker if possible.
    if trump_played or joker_played:
        trump_or_joker_cards = {
            card for card in hand
            if card.is_joker or (not card.is_joker and card.suit == state.trump)
        }
        if trump_or_joker_cards:
            return trump_or_joker_cards
        return set(hand)

    # If no trump or joker played, check the lead card for follow-suit rules
    lead_card = state.current_trick.plays[0].card

    # If a non-trump, non-joker suit was led, can follow suit or trump in.
    if (
        not lead_card.is_joker
        and state.trump is not None
        and lead_card.suit != state.trump
    ):
        same_suit_cards = {
            card for card in hand
            if not card.is_joker and card.suit == lead_card.suit
        }
        trump_or_joker_cards = {
            card for card in hand
            if card.is_joker or (not card.is_joker and card.suit == state.trump)
        }
        legal_plays = same_suit_cards | trump_or_joker_cards
        if legal_plays:
            return legal_plays
        return set(hand)

    # If a joker was led and is the only card played, anything may be played.
    return set(hand)


def get_trick_winner(trick: TrickState) -> Player:
    """ Return the winner of a terminal trick

    Preconditions: 
        - trick.trump is not None
    """

    if not trick.is_terminal:
        raise ValueError(f"trick is not in terminal state")

    trump_plays = list()
    trick_trump = None
    trick_trump_plays = list()
    joker_plays = list()

    for play in trick.plays:
        # Check if card is trump suit (excluding jokers)
        if not play.card.is_joker and play.card.suit == trick.trump:
            trump_plays.append(play)

        # Set trick trump to first non-joker, non-trump suit
        if not play.card.is_joker and play.card.suit != trick.trump and trick_trump is None:
            trick_trump = play.card.suit

        # Add to trick_trump_plays if it matches trick trump
        if not play.card.is_joker and play.card.suit == trick_trump:
            trick_trump_plays.append(play)

        # Track jokers
        if play.card.is_joker:
            joker_plays.append(play)

    # Trump suit beats everything
    if len(trump_plays) != 0:
        m = get_max_card([play.card for play in trump_plays])
        for play in trump_plays:
            if play.card == m:
                return play.player

    # Jokers beat sub-round trump
    elif len(joker_plays) != 0:
        return joker_plays[0].player

    # Sub-round trump (suit of first card) beats nothing
    else:
        m = get_max_card([play.card for play in trick_trump_plays])
        for play in trick_trump_plays:
            if play.card == m:
                return play.player
