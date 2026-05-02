# smear

## Abstract

Smear is a Canadian prairie card game that I was introduced to by my father's side of the family. It can be played with 3–8 people, but is optimally played with 4. [This](<https://en.wikipedia.org/wiki/Smear_(card_game)>) Wikipedia article and [this](https://gamerules.com/rules/canadian-smear/) GameRules article outline the general rules of the game; however, the implementation in this repository follows a specific house-ruled variant. The Rules section below documents that implemented ruleset exactly. This report attempts to find and describe optimal play based on a [minimax](https://en.wikipedia.org/wiki/Minimax) algorithm. Given the relative lack of popularity of the game (at least compared to other solved games like chess), there is not a large—or frankly any-sized—dataset of games played, which made analysis of the model difficult.

## Rules

The rules below describe the game as it is currently implemented in this repository. Where family or regional variants differ, treat this section as the source of truth for the engine.

### Setup

- The engine supports 3-8 players.
- Each player is dealt 6 cards every round.
- The full deck is 52 standard cards plus 2 jokers.
- The match target score is 21.
- The engine supports arbitrary team groupings, although a 4-player 2v2 game is the most natural version of smear.
- Card rank order is `2 < 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A`.

### Functional Deck and Hiding Cards

The game does not always use the full 54-card deck. Instead it builds a **functional deck** by removing all cards below a chosen rank, called the deck's **low**, so that the number of undealt hiding cards is as close to 2 as possible after dealing 6 cards per player.

The current implementation picks the following low by player count:

| Players | Low | Hiding cards |
| --- | --- | --- |
| 3 | 10 | 4 |
| 4 | 9 | 2 |
| 5 | 7 | 4 |
| 6 | 6 | 2 |
| 7 | 4 | 4 |
| 8 | 3 | 2 |

The deck is shuffled fresh every round. After dealing, any remaining cards are hidden and are not part of anyone's hand, but they still matter when deciding whether a trump jack exists and which trump cards count as high and low.

### Round Structure

Each round has two phases:

1. An auction to decide who leads the round.
2. Six tricks of card play, one per card in hand.

The winner of the auction leads the first card of the round. The winner of each trick leads the next trick.

### Auction

- The dealer rotates one seat clockwise each round.
- Bidding starts with the player to the dealer's left and proceeds clockwise, with the dealer acting last.
- Legal bids are integers from 1 through 6.
- Bids must strictly increase the current highest bid. Ties are not allowed.
- The auction is a single lap around the table: each player acts exactly once.
- If at least one player has already bid, later players may either overbid or pass.
- If nobody has bid yet, players may pass until the final player in the order. The final player is not allowed to pass; they must open the bidding with some value from 1 through 6.
- A bid of 6 does not end the auction early. Remaining players still get their one chance to act, but if 6 is already high their only legal move is to pass.
- After every player has acted once, the highest bidder wins the auction and leads the first trick.

### Trump and Leading the Round

- Trump is not chosen in a separate declaration step.
- Instead, the first card led in the round determines trump: the suit of that card becomes the round's trump suit.
- Because trump is not set until that first lead, the opening leader may not play a joker as the first card of the round.
- Once trump has been established, later trick leaders may lead any card in hand, including jokers.

### Trick Resolution

Each trick is won using the following priority:

1. Highest card in the round trump suit.
2. If no trump suit card was played, the first joker played.
3. If neither trump nor jokers were played, the highest card in the suit that was first led in the trick.

Additional details:

- Within a suit, higher rank wins using the normal rank order up to Ace.
- If multiple jokers are played and no trump suit card appears, the first joker played wins the trick.
- The trick winner captures every card in the trick, with one scoring exception described under low.

### Legal Plays During a Trick

The play restrictions in the engine are:

- If you are leading a trick after trump has already been established, any card is legal.
- If a trump suit card or a joker has already been played into the current trick, every later player must play either a trump suit card or a joker if they have one. If they have neither, they may play any card.
- If no trump suit card or joker has yet been played and the lead card is a non-trump suit card, a later player must play one of the following if possible:
  - a card in the led suit
  - a trump suit card
  - a joker
- Only if the player has none of those options may they discard any other card.

This means the engine allows trumping in or playing a joker on a non-trump lead, and once a trump suit card or joker appears, the remainder of the trick is forced into the trump-or-joker rule.

### Round Scoring

At the end of the round, up to 6 raw points are available:

- `high`: 1 point for the scoring unit that possesses the highest visible trump card in the functional deck.
- `jack`: 1 point for the scoring unit that possesses the jack of trump, if that jack is not hidden.
- `low`: 1 point for the scoring unit that was originally dealt the lowest visible trump card in the functional deck.
- `jokers`: 1 point for each joker possessed by the scoring unit.
- `game`: 1 point for the unique highest total of game-value cards captured.

Important scoring details:

- "Visible trump" means trump cards that exist in the functional deck and are not among the hidden undealt cards.
- If the jack of trump is hidden, nobody gets the jack point.
- Low is special: even if another player captures the trick containing low, the low point still belongs to the player or team that originally played that lowest visible trump card.
- Game-value totals are computed from captured cards using `10 = 10`, `J = 1`, `Q = 2`, `K = 3`, `A = 4`, and `2-9 = 0`.
- The game point is awarded only for a unique highest total. If the highest game total is tied, nobody receives the game point.

### Match Scoring

After raw round points are computed, the auction winner's bid is checked:

- If the auction winner's scoring unit made its bid, it adds its full round point total to its match score.
- If the auction winner's scoring unit failed to make its bid, it loses match points equal to the amount bid, not merely the amount it scored in the round.
- Every non-bidding scoring unit adds its raw round point total to its match score.

The implementation also applies one important match rule:

- Non-bidding scoring units are capped at `target_score - 1`, which is 20 in a standard game to 21.

In other words, only the scoring unit that won the auction can win the match at the end of a round. Everyone else can improve their score, but they cannot cross the finish line unless they were the bidder for that round.

## Local Debug Harness

This repo now includes a minimal local full-stack debug harness:

- `backend/` exposes a small FastAPI API around the existing Smear engine.
- `frontend/` is a plain React + Vite + TypeScript debug UI for inspecting and playing a round.

This is intentionally a basic testing interface, not the final product.

### Backend

Create a virtual environment, install the backend requirements, and run the API:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.server:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

Useful endpoints:

- `GET /health`
- `POST /game/new`
- `POST /game/auction/bid`
- `POST /game/auction/pass`
- `POST /game/reset` for a debug reset of the current round without advancing the match
- `POST /game/next-round` to advance the scored match to the next round
- `GET /game/state`
- `GET /game/legal-actions`
- `POST /game/play`
- `GET /game/score`

### Simulator

You can also run repeated all-bot matches from the command line:

```bash
python -m backend.simulator 1000 50 greedy random
```

To simulate same-model teams, set `--team-size`. For example, this runs a 2v2 match with two greedy bots on one team and two random bots on the other:

```bash
python -m backend.simulator --team-size 2 1000 50 greedy random
```

The simulator still caps games at 8 total seats, so `--team-size 2` allows up to four supplied models.

To override minimax search depth without changing bot ids, set `--depth`. For example, this runs the human-information and omniscient minimax bots at a 3-trick search depth:

```bash
python -m backend.simulator --depth 3 1000 50 one-trick-minmax o-one-trick-minmax
```

The simulator reports the applied override as `minimax_depth` in its JSON output.

Simulator output now includes wall-clock timing metrics such as `elapsed_seconds`, `average_seconds_per_game`, `average_seconds_per_round`, `games_per_second`, and `rounds_per_second` so you can compare algorithm changes directly.

### Frontend

In a second terminal, you can either install and run the frontend from the repo root:

```bash
npm install
npm run dev
```

Or run it directly from the frontend workspace:

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at [http://127.0.0.1:5173](http://127.0.0.1:5173).

By default the frontend talks to `http://127.0.0.1:8000`. If you need a different backend URL, set `VITE_API_BASE_URL` before starting Vite.
