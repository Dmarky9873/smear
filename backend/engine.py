from copy import deepcopy
from dataclasses import dataclass

try:
    from .models import (
        AuctionEvent,
        AuctionState,
        Player,
        Team,
        Card,
        Deck,
        RoundState,
        TrickState,
        Play,
        get_max_card,
    )
    from .constants import RANKS, HAND_SIZE, GAME_VALUES, RANK_ORDER
except ImportError:
    from models import (
        AuctionEvent,
        AuctionState,
        Player,
        Team,
        Card,
        Deck,
        RoundState,
        TrickState,
        Play,
        get_max_card,
    )
    from constants import RANKS, HAND_SIZE, GAME_VALUES, RANK_ORDER


@dataclass
class SearchTrickUndo:
    previous_current_player: Player
    previous_trump: str | None
    previous_trick_trump: str | None
    trick: TrickState
    acting_player: Player
    played_card: Card
    completed_trick: bool
    next_trick: TrickState | None = None
    trick_winner: Player | None = None


@dataclass
class SearchAuctionUndo:
    previous_current_bidder_index: int
    previous_highest_bid: int | None
    previous_highest_bidder_name: str | None
    previous_is_complete: bool
    previous_passed: bool
    action: AuctionEvent


class Auction:
    def __init__(
        self,
        player_names: list[str],
        dealer: str,
        max_bid: int = 6,
        auction_state: AuctionState | None = None,
    ):
        self._player_names = player_names
        self._max_bid = max_bid
        if auction_state is None:
            dealer_index = player_names.index(dealer)
            self._auction_state = AuctionState(
                dealer_index=dealer_index,
                current_bidder_index=(dealer_index + 1) % len(player_names),
                player_names=player_names,
            )
        else:
            self._auction_state = auction_state

    @classmethod
    def from_state(
        cls,
        auction_state: AuctionState,
        max_bid: int = 6,
    ) -> "Auction":
        return cls(
            player_names=auction_state.player_names,
            dealer=auction_state.dealer_name,
            max_bid=max_bid,
            auction_state=auction_state,
        )

    @property
    def state(self) -> AuctionState:
        return self._auction_state

    @property
    def current_bidder_name(self) -> str:
        return self._auction_state.current_bidder_name

    def legal_bid_amounts(self) -> list[int]:
        if self._auction_state.is_complete:
            return []
        minimum_bid = (
            1 if self._auction_state.highest_bid is None
            else self._auction_state.highest_bid + 1
        )
        if minimum_bid > self._max_bid:
            return []
        return list(range(minimum_bid, self._max_bid + 1))

    def can_pass(self) -> bool:
        if self._auction_state.is_complete:
            return False
        if self._auction_state.highest_bidder_name is not None:
            return True
        return len(self._auction_state.bid_history) < len(self._player_names) - 1

    def legal_actions(self) -> list[AuctionEvent]:
        bidder_name = self.current_bidder_name
        actions = [
            AuctionEvent(bidder_name=bidder_name, action="bid", amount=amount)
            for amount in self.legal_bid_amounts()
        ]
        if self.can_pass():
            actions.append(AuctionEvent(
                bidder_name=bidder_name, action="pass"))
        return actions

    def _next_bidder_index(self, start_index: int) -> int:
        return (start_index + 1) % len(self._player_names)

    def _finalize_if_needed(self) -> None:
        if len(self._auction_state.bid_history) < len(self._player_names):
            return
        if self._auction_state.highest_bidder_name is None:
            raise ValueError("auction cannot complete without a winning bid")
        self._auction_state.is_complete = True
        self._auction_state.current_bidder_index = self._player_names.index(
            self._auction_state.highest_bidder_name
        )

    def apply_event(self, event: AuctionEvent) -> AuctionState:
        if self._auction_state.is_complete:
            raise ValueError("auction is already complete")
        if event.bidder_name != self.current_bidder_name:
            raise ValueError("bidder passed is not the current bidder")

        if event.action == "pass":
            if not self.can_pass():
                raise ValueError(
                    "at least one player must bid before the auction can end"
                )
            self._auction_state.passed_player_names.add(event.bidder_name)
        elif event.action == "bid":
            if event.amount is None:
                raise ValueError("bid events require an amount")
            legal_amounts = self.legal_bid_amounts()
            if event.amount not in legal_amounts:
                raise ValueError(
                    f"illegal bid amount {event.amount}; legal bids are {legal_amounts}"
                )
            self._auction_state.highest_bid = event.amount
            self._auction_state.highest_bidder_name = event.bidder_name
        else:
            raise ValueError(f"unsupported auction action: {event.action}")

        self._auction_state.bid_history.append(event)
        self._finalize_if_needed()
        if not self._auction_state.is_complete:
            self._auction_state.current_bidder_index = self._next_bidder_index(
                self._auction_state.current_bidder_index
            )
        return self._auction_state


def get_legal_auction_actions(
    auction_state: AuctionState,
    max_bid: int = 6,
) -> list[AuctionEvent]:
    return Auction.from_state(auction_state, max_bid=max_bid).legal_actions()


def apply_auction_action_for_search(
    state: AuctionState,
    action: AuctionEvent,
    *,
    max_bid: int = 6,
    validate_legal: bool = True,
) -> SearchAuctionUndo:
    if state.is_complete:
        raise ValueError("auction is already complete")
    if action.bidder_name != state.current_bidder_name:
        raise ValueError("bidder passed is not the current bidder")

    if validate_legal and action not in get_legal_auction_actions(state, max_bid=max_bid):
        raise ValueError("auction action attempted despite it not being legal")

    undo = SearchAuctionUndo(
        previous_current_bidder_index=state.current_bidder_index,
        previous_highest_bid=state.highest_bid,
        previous_highest_bidder_name=state.highest_bidder_name,
        previous_is_complete=state.is_complete,
        previous_passed=action.bidder_name in state.passed_player_names,
        action=action,
    )

    if action.action == "pass":
        state.passed_player_names.add(action.bidder_name)
    elif action.action == "bid":
        if action.amount is None:
            raise ValueError("bid events require an amount")
        state.highest_bid = action.amount
        state.highest_bidder_name = action.bidder_name
    else:
        raise ValueError(f"unsupported auction action: {action.action}")

    state.bid_history.append(action)

    if len(state.bid_history) >= len(state.player_names):
        if state.highest_bidder_name is None:
            raise ValueError("auction cannot complete without a winning bid")
        state.is_complete = True
        state.current_bidder_index = state.player_names.index(state.highest_bidder_name)
    else:
        state.current_bidder_index = (state.current_bidder_index + 1) % len(state.player_names)

    return undo


def undo_auction_action_for_search(
    state: AuctionState,
    undo: SearchAuctionUndo,
) -> None:
    if not state.bid_history:
        raise ValueError("search undo expected an auction event in history")
    last_action = state.bid_history.pop()
    if last_action != undo.action:
        raise ValueError("search undo expected the applied auction action at history tail")

    if undo.action.action == "pass" and not undo.previous_passed:
        state.passed_player_names.remove(undo.action.bidder_name)

    state.current_bidder_index = undo.previous_current_bidder_index
    state.highest_bid = undo.previous_highest_bid
    state.highest_bidder_name = undo.previous_highest_bidder_name
    state.is_complete = undo.previous_is_complete


class Game:

    def __init__(
        self,
        num_players: int,
        player_names: list[str],
        teams,
        starting_player_name: str | None = None,
    ):

        if num_players > 8 or num_players < 3:
            raise ValueError(
                f"the number of players must be in [3, 8], you gave {num_players}")

        self._low, self._num_hiding, self._num_dealt = self._calculate_low(
            num_players)

        if len(player_names) != num_players:
            raise ValueError(
                f"you did not provide an adaquate number of names")

        players = [Player(name, set()) for name in player_names]
        player_dict = dict()
        for player in players:
            player_dict[player.name] = player

        teams_to_add = []
        for team in teams:
            new_team = Team([player_dict[name] for name in team], set())
            teams_to_add.append(new_team)

        if starting_player_name is None:
            starting_player = players[0]
        else:
            starting_player = player_dict.get(starting_player_name)
            if starting_player is None:
                raise ValueError(
                    f"starting player {starting_player_name} is not part of this game"
                )

        first_trick = TrickState(starting_player, [], players, None)

        self._round_state = RoundState(
            players,
            starting_player,
            None,
            first_trick,
            set(),
            [],
            teams_to_add,
            Deck(self._low),
        )

        self.reset_round(starting_player.name)

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

    @property
    def round_state(self) -> RoundState:
        return self._round_state

    @property
    def num_players(self) -> int:
        return len(self._round_state.players)

    def get_player_by_name(self, player_name: str) -> Player:
        for player in self._round_state.players:
            if player.name == player_name:
                return player
        raise ValueError(f"player {player_name} is not part of this game")

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

        return _apply_trick_action_in_place(self._round_state, action)

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

    def reset_round(self, starting_player_name: str | None = None) -> None:
        players = self._round_state.players
        for team in self._round_state.teams:
            team.captured_cards.clear()

        if starting_player_name is None:
            starting_player = players[0]
        else:
            starting_player = self.get_player_by_name(starting_player_name)

        first_trick = TrickState(starting_player, [], players, None)

        self._round_state = RoundState(
            players,
            starting_player,
            None,
            first_trick,
            set(),
            [],
            self._round_state.teams,
            Deck(self._low),
        )
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

    # If a non-trump, non-joker suit was led, players must follow suit if able.
    # Otherwise they may slough any card; simply holding trump or a joker does
    # not force or authorize a trump-in on this ruleset.
    if (
        not lead_card.is_joker
        and state.trump is not None
        and lead_card.suit != state.trump
    ):
        same_suit_cards = {
            card for card in hand
            if not card.is_joker and card.suit == lead_card.suit
        }
        if same_suit_cards:
            return same_suit_cards
        return set(hand)

    # If a joker was led and is the only card played, anything may be played.
    return set(hand)


def _apply_trick_action_in_place(state: RoundState, action: Play) -> bool:
    """Mutate a round state by applying a legal play.

    Returns whether the action completed the current trick.
    """
    if action.player.name != state.current_player.name:
        raise ValueError(
            f"action with player {action.player} was attempted despite {state.current_player} being the current player"
        )

    if action.card not in get_legal_actions(state):
        raise ValueError(
            f"action with card {action.card} attempted despite card not being in legal moves"
        )

    curr_player = state.current_player
    curr_trick = state.current_trick
    applied_action = Play(curr_player, action.card)
    curr_player.play_card(action.card)
    curr_trick.plays.append(applied_action)

    # If this is the first play of the entire round, set trump to the suit of this card.
    if len(state.trick_history) == 0 and len(curr_trick.plays) == 1:
        if not action.card.is_joker:
            curr_trick.trump = action.card.suit
            state.trump = action.card.suit

    players = state.players

    if curr_trick.is_terminal:
        trick_winner = get_trick_winner(curr_trick)
        state.current_player = trick_winner

        for play in curr_trick.plays:
            trick_winner.capture(play)

        state.current_trick = TrickState(
            trick_winner, [], curr_trick.players, state.trump
        )
        state.trick_history.append(curr_trick)
        return True

    state.current_player = players[(players.index(curr_player) + 1) % len(players)]
    return False


def apply_trick_action_for_search(
    state: RoundState,
    action: Play,
    *,
    validate_legal: bool = True,
) -> SearchTrickUndo:
    """Mutate a round state for search and return an undo record.

    Unlike `apply_trick_action_to_state`, this does not clone the state.
    Call `undo_trick_action_for_search` after exploring the child node.
    """
    if action.player.name != state.current_player.name:
        raise ValueError(
            f"action with player {action.player} was attempted despite {state.current_player} being the current player"
        )

    if validate_legal and action.card not in get_legal_actions(state):
        raise ValueError(
            f"action with card {action.card} attempted despite card not being in legal moves"
        )

    curr_player = state.current_player
    curr_trick = state.current_trick
    undo = SearchTrickUndo(
        previous_current_player=curr_player,
        previous_trump=state.trump,
        previous_trick_trump=curr_trick.trump,
        trick=curr_trick,
        acting_player=curr_player,
        played_card=action.card,
        completed_trick=False,
    )

    applied_action = Play(curr_player, action.card)
    curr_player.play_card(action.card)
    curr_trick.plays.append(applied_action)

    if len(state.trick_history) == 0 and len(curr_trick.plays) == 1:
        if not action.card.is_joker:
            curr_trick.trump = action.card.suit
            state.trump = action.card.suit

    players = state.players

    if curr_trick.is_terminal:
        trick_winner = get_trick_winner(curr_trick)
        undo.completed_trick = True
        undo.trick_winner = trick_winner
        state.current_player = trick_winner

        for play in curr_trick.plays:
            trick_winner.capture(play)

        next_trick = TrickState(
            trick_winner, [], curr_trick.players, state.trump
        )
        state.current_trick = next_trick
        state.trick_history.append(curr_trick)
        undo.next_trick = next_trick
        return undo

    state.current_player = players[(players.index(curr_player) + 1) % len(players)]
    return undo


def undo_trick_action_for_search(
    state: RoundState,
    undo: SearchTrickUndo,
) -> None:
    """Undo a search mutation created by `apply_trick_action_for_search`."""
    if undo.completed_trick:
        if not state.trick_history or state.trick_history[-1] is not undo.trick:
            raise ValueError("search undo expected the completed trick at history tail")
        state.trick_history.pop()

        if undo.trick_winner is None:
            raise ValueError("search undo is missing the trick winner")

        for play in undo.trick.plays:
            undo.trick_winner.captured_plays.remove(play)

    if not undo.trick.plays or undo.trick.plays[-1].card != undo.played_card:
        raise ValueError("search undo expected the played card at the trick tail")
    undo.trick.plays.pop()

    undo.acting_player.cards.add(undo.played_card)
    undo.trick.trump = undo.previous_trick_trump
    state.current_trick = undo.trick
    state.current_player = undo.previous_current_player
    state.trump = undo.previous_trump


def apply_trick_action_to_state(round_state: RoundState, action: Play) -> RoundState:
    """Return a new round state with the action applied.

    The input state is left unchanged.
    """
    new_state = deepcopy(round_state)
    _apply_trick_action_in_place(new_state, action)
    return new_state


def score_round(round: RoundState) -> dict[str, int]:
    details = score_round_details(round)
    return {
        result["name"]: result["total_points"]
        for result in details["results"]
    }


def score_round_details(round: RoundState) -> dict:
    if not round.is_terminal:
        raise ValueError("round is not in terminal state")

    if round.trump is None:
        raise ValueError("round trump has not been set")

    player_to_unit: dict[str, str] = {}
    scoring_units: list[dict] = []
    for team in round.teams:
        member_names = [player.name for player in team.constituents]
        unit_name = " / ".join(member_names)
        captured_cards = {
            card
            for player in team.constituents
            for card in player.captured_cards
        }
        scoring_units.append(
            {
                "name": unit_name,
                "member_names": member_names,
                "captured_cards": set(captured_cards),
                "breakdown": {
                    "high": 0,
                    "jack": 0,
                    "low": 0,
                    "jokers": 0,
                    "game": 0,
                },
                "joker_count": 0,
                "game_total": 0,
                "total_points": 0,
            }
        )
        for name in member_names:
            player_to_unit[name] = unit_name

    if len(scoring_units) != len(round.teams):
        raise ValueError("failed to construct scoring units")

    unit_by_name = {unit["name"]: unit for unit in scoring_units}

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
    jack_card = next(
        (card for card in visible_trump_cards if card.rank == "J"),
        None,
    )

    # low belongs to whoever was originally dealt / played it
    low_owner = None
    for play in all_completed_plays:
        if play.card == low_card:
            low_owner = play.player.name
            break

    if low_owner is None:
        raise ValueError(
            "could not determine owner of low card from completed plays")

    low_unit_name = player_to_unit[low_owner]

    # Low awards a point to the original owner, but the captured-card ownership
    # must remain untouched so game totals still reflect who actually won the card.
    unit_by_name[low_unit_name]["breakdown"]["low"] = 1

    # high goes to whoever possesses the highest visible trump
    high_unit_name = None
    for unit in scoring_units:
        if high_card in unit["captured_cards"]:
            high_unit_name = unit["name"]
            break

    if high_unit_name is None:
        raise ValueError(
            "could not determine owner of high card from scoring units")

    unit_by_name[high_unit_name]["breakdown"]["high"] = 1

    jack_unit_name = None
    jack_reason = None
    if jack_card is not None:
        for unit in scoring_units:
            if jack_card in unit["captured_cards"]:
                jack_unit_name = unit["name"]
                break

        if jack_unit_name is None:
            raise ValueError(
                "could not determine owner of jack card from scoring units")

        unit_by_name[jack_unit_name]["breakdown"]["jack"] = 1
    else:
        jack_reason = "Jack of trump is hiding, so no jack point is awarded."

    # one point per joker possessed
    for unit in scoring_units:
        joker_count = sum(
            1 for card in unit["captured_cards"] if card.is_joker)
        unit["joker_count"] = joker_count
        unit["breakdown"]["jokers"] = joker_count

    # game point: unique highest total only
    game_totals = {
        unit["name"]: sum(
            GAME_VALUES.get(card.rank, 0)
            for card in unit["captured_cards"]
        )
        for unit in scoring_units
    }

    game_winner_name = None
    tied_winners: list[str] = []
    max_total = 0
    if game_totals:
        max_total = max(game_totals.values())
        winners = [
            unit_name for unit_name, total in game_totals.items()
            if total == max_total
        ]
        if len(winners) == 1 and max_total > 0:
            game_winner_name = winners[0]
            unit_by_name[game_winner_name]["breakdown"]["game"] = 1
        else:
            tied_winners = winners

    for unit in scoring_units:
        unit["game_total"] = game_totals[unit["name"]]
        unit["total_points"] = sum(unit["breakdown"].values())

    return {
        "trump": round.trump,
        "high_card": high_card,
        "low_card": low_card,
        "awards": {
            "high": {
                "unit_name": high_unit_name,
                "card": high_card,
            },
            "jack": {
                "unit_name": jack_unit_name,
                "card": jack_card,
                "reason": jack_reason,
            },
            "low": {
                "unit_name": low_unit_name,
                "player_name": low_owner,
                "card": low_card,
            },
            "game": {
                "unit_name": game_winner_name,
                "game_total": max_total,
                "tied_unit_names": tied_winners,
                "reason": (
                    "Game total is tied, so no game point is awarded."
                    if game_winner_name is None and tied_winners
                    else None
                ),
            },
        },
        "results": scoring_units,
    }


def get_trick_winner(trick: TrickState) -> Player:
    """ Return the winner of a terminal trick

    Preconditions: 
        - trick.trump is not None
    """

    if not trick.is_terminal:
        raise ValueError(f"trick is not in terminal state")

    best_trump_play = None
    best_trump_rank = -1
    first_joker_play = None
    lead_suit = None
    best_lead_play = None
    best_lead_rank = -1

    for play in trick.plays:
        card = play.card
        if card.is_joker:
            if first_joker_play is None:
                first_joker_play = play
            continue

        card_rank = RANK_ORDER[card.rank]
        if card.suit == trick.trump:
            if card_rank > best_trump_rank:
                best_trump_rank = card_rank
                best_trump_play = play
            continue

        if lead_suit is None:
            lead_suit = card.suit
        if card.suit == lead_suit and card_rank > best_lead_rank:
            best_lead_rank = card_rank
            best_lead_play = play

    if best_trump_play is not None:
        return best_trump_play.player
    if first_joker_play is not None:
        return first_joker_play.player
    if best_lead_play is not None:
        return best_lead_play.player
    raise ValueError("terminal trick did not contain a winning play")
