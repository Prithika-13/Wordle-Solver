# Wordle solver: information theory vs. reinforcement learning

Two ways to play Wordle, compared head to head on the same word lists:

1. **Entropy solver** – greedily plays the guess that maximises the expected information
   (bits) about the hidden answer.
2. **RL agent** – a PPO-trained policy that scores candidate guesses with a small shared MLP.
   Two variants: **blind** (only letter statistics as features) and **assisted** (also gets
   entropy features).

Data: 2,309 answers, 12,953 allowed guesses (original Wordle lists). Failures (not solved in
6 guesses) count as 7 guesses in the averages.

## Results

Held-out evaluation: answers are split 80/20 with a fixed seed; the RL agents never saw the 461
test answers as targets. RL rows are the mean of 3 training seeds.

| Solver | Mean guesses (test) | Solve rate | Notes |
|---|---|---|---|
| Random (consistent word) | 4.10 | 96.7% | |
| Letter-frequency heuristic | 3.66 | 99.1% | |
| **RL blind** | **3.66** (3.657–3.668) | 99.6% | matches the heuristic, no look-ahead |
| Entropy, candidates only | 3.58 | 99.6% | |
| **RL assisted** | **3.51** (3.497–3.514) | 100% | ties the entropy solver |
| Entropy, all guesses | **3.50** | 100% | best |

On all 2,309 answers the entropy solver averages **3.463** (opener `soare`, 100% solved, max 6).

![comparison](results/comparison.png)
![learning curves](results/learning_curves.png)

### What the numbers say
- **Entropy wins, or ties.** The learned policy with entropy features (3.505 mean over 3 seeds) is
  statistically indistinguishable from the entropy solver (3.501). Per answer it is better on
  170 words, worse on 169, and equal on 122, so there is no real difference.
- **Learning alone (blind) reaches the heuristic level, not the entropy level.** Without
  look-ahead features the agent plateaus at ~3.66, the same as a hand-written letter-frequency
  rule. All three seeds converged on the opener `stare`.
- **The assisted agent chose different openers** (`salet` for seeds 0 and 1, `raile` for
  seed 2) than the entropy solver (`soare`) yet scores the same, which shows that several
  openers are near-equivalent under the greedy criterion.
- **Hardest words** are analysed in the next section (look-alike clusters and rare letters).

### Caveats (read before quoting the numbers)
- The assisted agent chooses among a **shortlist of 48 words ranked by entropy** (40 top-entropy
  words + up to 8 candidates), so it is "RL on top of an entropy prior", not RL from scratch.
  The blind agent's shortlist is ranked by letter statistics.
- The entropy solver uses a simple finishing rule (guess a candidate when ≤2 remain); the RL
  agents do not get that rule.
- The word lists are the original 2,309-answer Wordle lists with a uniform prior, so the
  numbers are not comparable with published results that use word frequencies or the NYT list.
- 300 PPO iterations × 128 episodes (~38k games) per seed; a longer run might close the small
  remaining gap, but I did not test that.

## Interactive dashboard

`streamlit run app/streamlit_app.py` opens **Wordle Solver Lab** with seven tabs:

| Tab | What it does |
|---|---|
| 🎮 Solve a game | Enter your real guesses and colours (click a square to cycle gray/yellow/green). See what all four solvers would play next, with expected bits. |
| 🎓 Coach report | Grades every guess of your game (or a pasted one): **skill** (was it a good question?) versus **luck** (was the feedback kind?), with the best alternative for each guess and a replay of the entropy solver on the same answer. |
| 📈 My history | The coach's memory: the games you saved, with your average skill and luck, skill trend, openers, weakest guess number and repeated weak words. Download or delete it any time. |
| 🔎 Watch a word | Pick any answer and watch every solver play it side by side, with the information each guess revealed. |
| 📊 Results | Held-out comparison, guess distribution and the RL learning curves. |
| 🧠 Why words are hard | The difficulty analysis below, a top-20 table and a lookup for any word. |
| 📖 How it works | Short explanation of the two techniques and the limits. |

![play](docs/ui_play.png)
![coach](docs/ui_coach.png)
![history](docs/ui_history.png)
*(The history screenshot uses demo games played by a script, to show the layout.)*
![results](docs/ui_results.png)
![insights](docs/ui_insights.png)

## Coach report: skill vs. luck

For each guess the coach computes the **expected bits** the guess should reveal (entropy over the
words still possible) and compares it with the **best available** guess. The ratio is the guess's
*skill* score. The information the colours *actually* revealed, minus the expectation, is the
*luck*. Example: opening with `funny` scores 3.33 expected bits against 5.89 for `soare` (a "Weak"
guess), and it was also unlucky (-2.15 bits). Pass a finished game and the report also replays the
entropy solver on the same answer. Logic lives in `src/coach.py` (unit-tested); limits: it assumes
all remaining words are equally likely, so it can't judge guesses by word familiarity.

## Ask the coach (LLM with tool calling)

The Coach tab ends with **Ask the coach**, a Claude agent that answers questions such as *"Was I
unlucky, or did I play badly?"* or *"Why is WATCH so hard?"*. The LLM does **no arithmetic and has no
memory of the game**: it decides which tool to call, reads the exact result and explains it. Every
answer comes with a collapsible panel showing the tool calls and their raw outputs.

| Tool (`src/coach_tools.py`) | Wraps |
|---|---|
| `get_game_report` | the coach grader (`src/coach.py`): per-guess skill, luck, ratings, solver replay |
| `score_guess` | entropy scoring: how good is any word at any position of the game |
| `best_guesses` | top guesses by expected information at a position |
| `remaining_answers` | the candidate filter: how many and which answers are still possible |
| `solve_with_entropy_solver` | a full entropy-solver game against a given answer |
| `get_player_history` | the memory below: habits and progress across saved games |
| `word_difficulty` | the difficulty table from the analysis above |

- **Agent loop** (`src/coach_agent.py`): a manual tool-use loop on the Anthropic SDK. All tool results of a
  turn go back in one message, bad tool input comes back as an `is_error` result instead of crashing, and
  the loop is capped at 8 steps. Refusals, truncated answers and API errors are handled and shown.
- **Model:** `claude-opus-5-5` by default (override with `COACH_MODEL`). The request opts into the API's
  server-side refusal fallback for models that support it.
- **Setup:** put the key in a git-ignored `.env` file (copy `.env.example` to `.env`; the app loads it at
  startup, and an exported `ANTHROPIC_API_KEY` takes priority), or use a Streamlit secret, or paste a key
  into the tab; a pasted key is kept in memory for that browser session only. Without a key the section explains this and
  the rest of the app works unchanged. Your question and the tool results are sent to the Anthropic API.
- **Testing:** the tools, the loop, the summary and its grounding check are unit-tested against a scripted fake client (tool-result message
  format, error handling, request parameters, history, retry on ungrounded figures). I could not call the live API from the build
  environment, so the end-to-end behaviour with a real model is untested; try a few questions and check the
  tool panel against the report above.

## AI coach summary (LLM-written, programmatically verified)

Press **✍️ Write AI summary** in the Coach tab and Claude writes a short, personal summary: a headline, what went
well, what to improve and one drill. It is generated as **structured output** (a fixed JSON schema, validated with
Pydantic) from facts that code has already computed. Nothing is calculated by the model, and a checker
(`src/coach_summary.py`) then verifies the result:

- every **figure** it cites must exist in the facts (allowing honest rounding, e.g. 5.885 written as 5.89, and 0.71 written as 71%);
- every **capitalised word** must be a word from the game or the solver's analysis;
- if something fails, the model is shown the problems and asked **once** to rewrite; anything still unverified is
  shown to you as a warning instead of being hidden.

Known limits of the check: integers 0-10 are always allowed (they are mostly ordinals such as "guess 3"), and it
verifies that a figure exists in the facts, not that it is attached to the right guess. The template verdict above
the card is kept as a deterministic fallback and works without an API key.

## Memory: the coach remembers your games

Press **💾 Save to my history** on a finished game (solved, or all six guesses used; example games can't be
saved) and `src/memory.py` stores that game's grades. The coach then knows your habits across games, both on
the **My history** tab and through the `get_player_history` tool that Ask the coach and the AI summary can use.

- **What is stored:** only facts computed by the grader (guesses, colours, bits, skill, luck, ratings), never
  AI-written text. The coach therefore can't "remember" something that didn't happen, and stored text can't
  carry injected instructions.
- **Where:** a plain JSON file, `data/player_history.json` by default (git-ignored; override with
  `WORDLE_MEMORY_PATH`). Saving is explicit and idempotent (the same game is never stored twice), writes are
  atomic, and an unreadable file is kept as `*.corrupt.json` instead of being overwritten.
- **What it computes:** games saved, solve rate, average skill, luck and guesses, a skill trend (only from 6
  games), openers and how good each is, average quality per guess number, repeated weak words and recent games.
  The coach is instructed to say how many games a pattern rests on and not to call anything a habit from fewer
  than 3 games.
- **Control:** the tab has a JSON download and a delete button that needs a confirmation tick.

## Why are some words hard? (difficulty analysis)

`python scripts/difficulty.py` measures how many guesses each of the 2,309 answers needs
(average over the entropy solver and the three assisted RL agents) and fits interpretable
models on 8 word features.

![difficulty](results/difficulty.png)

- **Word features only explain a modest share of the difficulty.** Cross-validated R²:
  linear model **0.23**, depth-3 decision tree 0.13, random forest 0.25. The rest depends on
  which specific words survive each guess, which these features can't capture.
- **Clear drivers** (standardised effect, 95% bootstrap interval excludes zero): words with a
  **rare letter** (+0.10 guesses per standard deviation) or generally uncommon letters,
  **look-alike clusters** (+0.08), words that stay **crowded after the best opener** (+0.07) and
  **repeated letters** (+0.06). One-letter neighbours and positional letter frequency show no clear effect.
- **Look-alike clusters matter most at the extreme.** Words sharing 4 letters in place with
  7-9 other words (`_ATCH`, `_IGHT`, `_OUND`) average **3.95** guesses, versus 3.43 for words
  with no cluster. Only 38 words are in that bucket.
- **Hardest words:** `boxer` (5.75), then `watch`, `waver`, `joker` (5.25), then `wight`, `bound`,
  `cover`, `mover`, `bitty`, `baker`, `poker`, `frame`, `poppy` (5.00). Two kinds: crowded
  clusters (`watch`, `wight`, `bound`, `cover`) and rare or repeated letters (`boxer`, `joker`, `poppy`).
- **Caveats:** difficulty is measured for these solvers, not for human players, and the analysis is
  correlational.

## Project layout
```
data/        answers.txt, allowed.txt (+ cached feedback_matrix.npy, git-ignored, rebuilt on first run)
src/
  feedback.py      feedback with correct duplicate letters, vectorised matrix
  features.py      per-word features + shortlist for the RL agent
  difficulty.py    word-difficulty features (look-alike clusters, letter rarity, ...)
  coach.py         per-guess skill / luck grading of a game
  coach_tools.py   the 6 tools the coach LLM can call (pure, tested without an LLM)
  coach_agent.py   Claude tool-use loop behind "Ask the coach"
  coach_summary.py LLM-written summary (structured output) + grounding check
  memory.py        player memory: saved games and the statistics computed from them
  env.py           Gymnasium environment
  evaluate.py      run solvers over all answers
  solvers/         random, frequency, entropy, rl_agent
scripts/
  train_rl.py      PPO training    compare.py   held-out comparison    plots.py   figures
  difficulty.py    why are some words hard?  (results/difficulty.csv, difficulty_report.json)
app/streamlit_app.py   the 7-tab dashboard
tests/             pytest suite (feedback edge cases, solvers, env)
results/           CSVs, figures, trained policies (*.pt)
```

## Reproduce
```bash
pip install -r requirements.txt
make test
make baselines                                   # random / frequency / entropy (~30 s)
python scripts/train_rl.py --mode assisted --seed 0 --iters 300   # ~14 min on 1 CPU thread
python scripts/train_rl.py --mode blind    --seed 0 --iters 300   # ~3.5 min
python scripts/compare.py && python scripts/plots.py
python scripts/difficulty.py                     # difficulty analysis
make app                                         # dashboard (needs torch for the RL solvers)
```

## Design notes
- **Feedback** is a base-3 integer (0–242). Greens are assigned first, then yellows only up to
  the remaining count of each letter; this is the classic duplicate-letter bug and is unit-tested.
- **Feedback matrix** `[12953, 2309]` is precomputed once, so every entropy score is a table
  lookup plus a `bincount`.
- **Policy** is word-agnostic: one MLP scores each shortlisted word from its features, so it
  generalises to answers it never trained on (train and test scores are close).
- **Reward** is −1 per guess, −4 extra on failure, plus potential-based shaping
  (0.3 × bits of information gained), which does not change the optimal policy but gives
  a learning signal when a random policy almost never wins.

## Ideas to extend
Hard mode, an adversarial (Absurdle-style) opponent, two-step look-ahead, word-frequency priors,
and training the agent without any shortlist.
