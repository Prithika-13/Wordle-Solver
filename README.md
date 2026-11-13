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

`streamlit run app/streamlit_app.py` opens **Wordle Solver Lab** with five tabs:

| Tab | What it does |
|---|---|
| 🎮 Solve a game | Enter your real guesses and colours (click a square to cycle gray/yellow/green). See what all four solvers would play next, with expected bits. |
| 🔎 Watch a word | Pick any answer and watch every solver play it side by side, with the information each guess revealed. |
| 📊 Results | Held-out comparison, guess distribution and the RL learning curves. |
| 🧠 Why words are hard | The difficulty analysis below, a top-20 table and a lookup for any word. |
| 📖 How it works | Short explanation of the two techniques and the limits. |

![play](docs/ui_play.png)
![results](docs/ui_results.png)
![insights](docs/ui_insights.png)

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
  env.py           Gymnasium environment
  evaluate.py      run solvers over all answers
  solvers/         random, frequency, entropy, rl_agent
scripts/
  train_rl.py      PPO training    compare.py   held-out comparison    plots.py   figures
  difficulty.py    why are some words hard?  (results/difficulty.csv, difficulty_report.json)
app/streamlit_app.py   the 5-tab dashboard
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
