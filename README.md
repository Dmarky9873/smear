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
- If trump was led, any trump card or joker is legal.
- If trump was led and you have at least one trump card or joker, you may not discard a non-trump off-suit card.
- If a non-trump, non-joker card was led, you may always play a trump suit card or a joker.
- If you choose to play a non-trump, non-joker card, it must follow the led suit if possible.
- Only if the led card was not trump and you have no card in the led suit may you discard any non-trump, non-joker off-suit card.

This means trump cards and jokers are always legal during a trick, but a trump lead blocks non-trump discards while you still hold any trump-capable response.

Examples:

- Trump is hearts, the trick is led with `AD`, and your hand is `9D`, `10H`, `J1`, `KS`. Your legal plays are `9D`, `10H`, and `J1`. `KS` is illegal because it is neither trump, nor a joker, nor a diamond.
- Trump is hearts, the trick is led with `AH`, and your hand is `9D`, `10H`, `J1`, `KS`. Your legal plays are `10H` and `J1`. Because hearts were led, a non-trump, non-joker off-suit discard is not allowed while you still hold trump.
- Trump is diamonds, the trick is led with `AD`, and your hand is `JH`, `QH`, `AH`, `10S`, `KS`, `J2`. Your only legal play is `J2`. Because trump was led, the joker is the only trump-capable response in hand.
- If the trick is led with a joker, every card in hand is legal.

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

## Monorepo Apps

This repo now carries the browser clients and the engine in one monorepo-style layout:

- `backend/` exposes the FastAPI API around the existing Smear engine.
- `frontend/` is the internal debug UI. It still includes the inspector-heavy debug surface plus the older play mode.
- `apps/play-ui/` is the cleaner public-facing play app intended for online access.
- `packages/web-core/` contains the shared TypeScript game client, types, helpers, and card component used by both frontends.

The public app and the debug app both talk to the same backend API, but they now isolate games by browser session instead of sharing one global in-memory match.

## Local Debug Harness

This repo includes two local browser frontends:

- `npm run dev:play` starts the public play app.
- `npm run dev:debug` starts the internal debug app.

Both expect the FastAPI backend to be running locally.

### Backend

Create a virtual environment, install the backend requirements, and run the API:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.server:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

Useful environment variables for the backend:

- `SMEAR_CORS_ORIGINS` is a comma-separated allowlist of frontend origins. By default the API allows the two local Vite apps on ports `5173` and `5174`.
- `SMEAR_STATE_DB_PATH` controls where browser sessions are persisted. The default is `.smear/sessions.sqlite3`; set it to `none` to use memory only.
- `SMEAR_SESSION_TTL_HOURS` controls how long inactive browser sessions are kept in the in-memory cache. Persisted sessions can still be restored from `SMEAR_STATE_DB_PATH` after cache expiry or server restart.
- `STRIPE_SECRET_KEY` enables the donation checkout endpoint. Set it in your runtime environment or host secrets manager, not in source code.
- `SMEAR_PUBLIC_SITE_URL` is the browser URL Stripe should redirect back to after donation checkout, for example `https://play-smear.com`. Local dev falls back to the Vite play app URL.
- `SMEAR_DONATION_CURRENCY` controls donation currency. It defaults to `cad`.

Production packaging intentionally excludes neural bot replay/checkpoint output under `backend/bots/models/`. Only the small runtime bundles named `neural_3p_v*.json` should be committed or shipped to Railway.

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
- `POST /donations/checkout-session` creates a Stripe Checkout Session for a one-time site donation.
- `WS /game/ws?session_id=...` streams `game_state` messages for the session whenever the game changes.
- `POST /lobbies` creates a multiplayer lobby with a shareable code and host player token.
- `POST /lobbies/{code}/join` adds a player to an open lobby seat.
- `POST /lobbies/{code}/start` starts a full lobby as a shared human-controlled match.
- `WS /lobbies/{code}/ws?player_token=...` streams player-scoped lobby and game updates.
- `GET /learn/challenge?bot_id=optimal-bot` returns a filtered practice position, legal learner options, the selected bot action, and an explanation to reveal after the learner chooses.

Every stateful endpoint also accepts an `X-Smear-Session-Id` header. The browser apps generate and persist that header automatically so each browser gets its own isolated game.

### Frontends

Install the root workspace dependencies once:

```bash
npm install
```

Run the public play app:

```bash
npm run dev:play
```

Run the internal debug app:

```bash
npm run dev:debug
```

By default:

- the debug app runs at [http://127.0.0.1:5173](http://127.0.0.1:5173)
- the public play app runs at [http://127.0.0.1:5174](http://127.0.0.1:5174)

To point either frontend at a deployed backend, set `VITE_API_BASE_URL` before starting or building it.

For the donation page, the browser display currency defaults to CAD. Set `VITE_DONATION_CURRENCY` to match `SMEAR_DONATION_CURRENCY` if you change it.

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

To reduce variance from fixed seat order, set `--fair`. This rotates and reverses seat assignments across the supplied models, and reports the tested schedule in `fair_schedule`:

```bash
python -m backend.simulator --fair 60 50 o-3-trick-minmax greedy
```

For reproducible fair comparisons, add `--seed`. In fair mode, each batch of rotated seat assignments shares the same RNG seed so the models are compared on matched deals:

```bash
python -m backend.simulator --fair --seed 0 60 50 o-3-trick-minmax greedy
```

For a long-running console-only ladder across the ready bot pool, use `continuous-sim`:

```bash
python continuous-sim --duration-hours 8 --workers 4
```

By default it uses every visible ready bot, samples three distinct bots per game, runs those free-for-all matches continuously, and keeps a live console panel updated with the current Elo table plus compact per-match progress rows. The visible ready bot set is intentionally capped at the functional presets up through 3-trick depth; deeper minimax presets remain hidden because they are too slow for normal interactive use.

The default scheduler is `balanced`, not purely random. In balanced mode, `continuous-sim` cycles through every unique three-bot trio in the selected pool and rotates all seat orders before repeating, which makes the ladder much less sensitive to sampling noise and seat-position artifacts. The live panel shows the current balanced-cycle progress as `Schedule | balanced | cycle ...`.

When only two bots are selected, `continuous-sim` fills the third seat with a `random` player so the game can still run as a three-player match. That filler seat is shown in the live match rows, but the Elo table continues to track only the selected bot pool.

The Elo update is multiplayer-aware rather than winner-only:

- Each match compares every rated bot to every other rated bot in the same three-player game.
- Higher final match score at termination counts as a win for that pairwise comparison, lower score counts as a loss, and equal scores count as a draw.
- The total Elo delta for the match is the average of those pairwise updates, scaled by the configured `--k-factor` (default `32`).

By default `continuous-sim` persists ratings in `continuous-sim-elo.json` at the repo root and loads that file again on the next run, so later runs keep updating the same Elo history instead of starting over. Use `--fresh-ratings` if you want to ignore the saved ladder at startup and begin from `--initial-rating`, and use `--elo-file /path/to/file.json` to point at a different ladder file.

The leaderboard now includes a `c95` column, which is the approximate 95% confidence half-width for each rating. Smaller values mean the bot's position is more settled; if two bots' ratings are close and their confidence bands overlap heavily, treat them as the same tier rather than a decisive ordering.

Useful flags:

- `--games 100` stops after a fixed number of matches.
- `--duration-hours 8` is useful for longer unattended runs.
- `--workers 1` disables parallel workers and runs serially in the current process.
- `--bots random greedy stupid` restricts the pool to a chosen subset.
- `--include-hidden` also adds the hidden legacy bot ids to the default pool.
- `--elo-file /tmp/nightly-elo.json` saves and reloads Elo from a specific JSON file.
- `--fresh-ratings` starts from the configured initial rating instead of loading the saved Elo JSON first.
- `--schedule balanced` uses balanced trio and seat-order cycles; `--schedule random` restores simple random sampling.

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
