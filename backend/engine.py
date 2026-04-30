from models import Player, Team, Card, Deck, RoundState, TrickState, Play, get_max_card
from constants import RANKS, HAND_SIZE, GAME_VALUES, RANK_ORDER


class Game:

    def __init__(self, num_players: int, player_names: list[str], teams: set[tuple]):

        if num_players > 8 or num_players < 3:
            raise ValueError(
                f"the number of players must be in [3, 8], you gave {num_players}")

        self._low, self._num_hiding, self._num_dealt = self._calculate_low(
            num_players)

        print(f"creating a game with a low of {self._low}...")

        if len(player_names) != num_players:
            raise ValueError(
                f"you did not provide an adaquate number of names")

        players = [Player(name, set()) for name in player_names]
        player_dict = dict()
        for player in players:
            player_dict[player.name] = player

        curr_player = players[0]

        teams_to_add = []
        for team in teams:
            new_team = Team([player_dict[name] for name in team], set())
            teams_to_add.append(new_team)

        first_trick = TrickState(players[0], [], players, None)

        self._round_state = RoundState(players,
                                       curr_player, None, first_trick, set(), [], teams_to_add, Deck(self._low))
        self._deal_cards()

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

    def set_trump(self, trump) -> None:
        trick = self._round_state.current_trick
        round = self._round_state
        trick.trump = trump
        round.trump = trump

    @property
    def low(self):
        return self._low

    def set_starting_player(self, starting_player: Player) -> None:
        self._round_state.current_player = starting_player
        self._round_state.current_trick.leader = starting_player

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

        # If this is the first play of the entire round, set trump to the suit of this card
        if len(self._round_state.trick_history) == 0 and len(curr_trick.plays) == 1:
            if not action.card.is_joker:
                self.set_trump(action.card.suit)

        players = self._round_state.players

        if self._round_state.current_trick.is_terminal:

            trick_winner = get_trick_winner(curr_trick)

            self._round_state.current_player = trick_winner

            for play in curr_trick.plays:
                trick_winner.capture(play)

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

    def _deal_cards(self) -> None:
        """Shuffles and deals cards to the players within the game

        Returns:
            list[Card]: Cards that are hiding based on this shuffle
        """
        deck = Deck(self._low)
        deck.shuffle()
        remaining_cards = deck.get_copy()

        for player in self._round_state.players:
            player.receive_new_hand(
                {remaining_cards.pop() for _ in range(HAND_SIZE)})

        self._round_state.deck = deck
        self._round_state.hidden_cards = set(remaining_cards)

    def reset_round(self) -> None:
        players = self._round_state.players

        first_trick = TrickState(players[0], [], players, None)

        self._round_state = RoundState(players,
                                       players[0], None, first_trick, set(), [], self._round_state.teams, Deck(self._low))
        self._deal_cards()


def get_legal_actions(state: RoundState) -> set[Card]:
    hand = state.current_player.cards

    if not state.current_trick.plays:
        # First card of the trick
        if state.trump is None:
            # First player of the round cannot play a joker
            return {card for card in hand if not card.is_joker}
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


def score_round(round: RoundState) -> dict[str, int]:
    if not round.is_terminal:
        raise ValueError("round is not in terminal state")

    if round.trump is None:
        raise ValueError("round trump has not been set")

    player_points = {player.name: 0 for player in round.players}
    possessed_cards = {
        player.name: set(player.captured_cards) for player in round.players
    }

    all_completed_plays = [
        play for trick in round.trick_history for play in trick.plays
    ]

    visible_trump_cards = [
        card for card in round.deck.get_copy()
        if (
            not card.is_joker
            and card.suit == round.trump
            and card not in round.hidden_cards
        )
    ]

    if not visible_trump_cards:
        raise ValueError(
            "no non-hidden trump cards are available to score high/low")

    low_card = min(visible_trump_cards, key=lambda card: RANK_ORDER[card.rank])
    high_card = max(visible_trump_cards,
                    key=lambda card: RANK_ORDER[card.rank])

    # low belongs to whoever was originally dealt / played it
    low_owner = None
    for play in all_completed_plays:
        if play.card == low_card:
            low_owner = play.player.name
            break

    if low_owner is None:
        raise ValueError(
            "could not determine owner of low card from completed plays")

    # make sure low is credited to the original owner for scoring purposes
    for cards in possessed_cards.values():
        cards.discard(low_card)
    possessed_cards[low_owner].add(low_card)
    player_points[low_owner] += 1

    # high goes to whoever possesses the highest visible trump
    high_owner = None
    for player_name, cards in possessed_cards.items():
        if high_card in cards:
            high_owner = player_name
            break

    if high_owner is None:
        raise ValueError(
            "could not determine owner of high card from possessed cards")

    player_points[high_owner] += 1

    # one point per joker possessed
    for player_name, cards in possessed_cards.items():
        joker_count = sum(1 for card in cards if card.is_joker)
        player_points[player_name] += joker_count

    # game point: unique highest total only
    game_totals = {
        player_name: sum(GAME_VALUES.get(card.rank, 0) for card in cards)
        for player_name, cards in possessed_cards.items()
    }

    if game_totals:
        max_total = max(game_totals.values())
        winners = [
            player_name for player_name, total in game_totals.items()
            if total == max_total
        ]
        if len(winners) == 1 and max_total > 0:
            player_points[winners[0]] += 1

    return player_points


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
