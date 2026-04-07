# GAME CONFIG
HAND_SIZE = 6

# CARDS

CARD_DICT = {
    "AD": "Ace of Diamonds",
    "2D": "Two of Diamonds",
    "3D": "Three of Diamonds",
    "4D": "Four of Diamonds",
    "5D": "Five of Diamonds",
    "6D": "Six of Diamonds",
    "7D": "Seven of Diamonds",
    "8D": "Eight of Diamonds",
    "9D": "Nine of Diamonds",
    "10D": "Ten of Diamonds",
    "JD": "Jack of Diamonds",
    "QD": "Queen of Diamonds",
    "KD": "King of Diamonds",
    "AH": "Ace of Hearts",
    "2H": "Two of Hearts",
    "3H": "Three of Hearts",
    "4H": "Four of Hearts",
    "5H": "Five of Hearts",
    "6H": "Six of Hearts",
    "7H": "Seven of Hearts",
    "8H": "Eight of Hearts",
    "9H": "Nine of Hearts",
    "10H": "Ten of Hearts",
    "JH": "Jack of Hearts",
    "QH": "Queen of Hearts",
    "KH": "King of Hearts",
    "AS": "Ace of Spades",
    "2S": "Two of Spades",
    "3S": "Three of Spades",
    "4S": "Four of Spades",
    "5S": "Five of Spades",
    "6S": "Six of Spades",
    "7S": "Seven of Spades",
    "8S": "Eight of Spades",
    "9S": "Nine of Spades",
    "10S": "Ten of Spades",
    "JS": "Jack of Spades",
    "QS": "Queen of Spades",
    "KS": "King of Spades",
    "AC": "Ace of Clubs",
    "2C": "Two of Clubs",
    "3C": "Three of Clubs",
    "4C": "Four of Clubs",
    "5C": "Five of Clubs",
    "6C": "Six of Clubs",
    "7C": "Seven of Clubs",
    "8C": "Eight of Clubs",
    "9C": "Nine of Clubs",
    "10C": "Ten of Clubs",
    "JC": "Jack of Clubs",
    "QC": "Queen of Clubs",
    "KC": "King of Clubs",
    "J1": "Joker 1",
    "J2": "Joker 2",
}

SUITS = {
    "H": "Hearts",
    "D": "Diamonds",
    "C": "Clubs",
    "S": "Spades"
}

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_ORDER = {rank: i for i, rank in enumerate(RANKS, start=2)}
GAME_VALUES = {
    "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0,
    "8": 0, "9": 0, "10": 10, "J": 1, "Q": 2, "K": 3, "A": 4,
}
