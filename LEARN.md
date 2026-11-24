# LEARN.md: make this repo yours

Reading code you did not write teaches you very little. Interviewers can tell. This file turns the repo into practice: concept cards, exercises you implement, and questions you answer out loud with the code closed.

## How to use it

1. Do the exercises in `exercises/` (about 2 hours total). Run `pytest exercises -q`. Unimplemented ones show as skipped; exercise 5 fails until you fix the bug, on purpose.
2. Answer the 12 questions below in your own words, in writing, with no code open.
3. Paste your answers to Claude and ask it to find the gaps. Ask for a re-quiz a few days later on the ones you missed.

Do not copy the reference code in `src/` into the exercises. The point is to get stuck once.

## Exercises

| File | You implement | Concept |
|---|---|---|
| `ex1_elo_update.py` | One Elo update with the margin-of-victory multiplier | Rating systems, why upsets move ratings more |
| `ex2_ipw.py` | Normalised inverse-probability-weighted treatment effect | Confounding, propensity scores |
| `ex3_attention.py` | Scaled dot-product attention in PyTorch | What a transformer actually computes |
| `ex4_temperature.py` | Temperature scaling without LBFGS | Calibration |
| `ex5_find_the_leak.py` | Find and fix a one-word data leak | Leakage, the bug that ruins most sports models |

## Concept cards (real explanation, simple terms, memory hook)

**Leakage and the "rewrite the future" test**
- *Real:* A feature leaks if its value for a game depends on information not available before tipoff, including that game's own result. A model trained on a leaky feature learns the leak, scores brilliantly in testing, and fails in use. The reliable check is behavioural: change the outcome of the game (and everything after it), recompute, and assert the feature did not move.
- *Simple:* If you can change the answer and the question changes, the question was cheating.
- *Hook:* **Rewrite the future; the past must not flinch.**

**Baselines, controls, and decomposing a gain**
- *Real:* "Beats the benchmark" is not a finding until you know why. A control that isolates one candidate explanation (here: refit home-court advantage, nothing else) tells you how much of the gain was modelling and how much was fixing a known miscalibration.
- *Simple:* Before celebrating that you beat the old model, check whether you just fixed one number.
- *Hook:* **Beat the benchmark, then beat the lazy fix.**

**Calibration and temperature scaling**
- *Real:* A calibrated model's 70% predictions win about 70% of the time. Temperature scaling divides logits by one learned number T before the sigmoid. It rescales confidence without reordering predictions.
- *Simple:* A volume knob for confidence, not a different opinion.
- *Hook:* **T turns confidence up or down; it never changes who it picks.**

**Are clusters real?**
- *Real:* k-means always returns clusters. Silhouette measures separation, but only means something against a null (shuffle each column independently and rerun) and a stability check (do held-out points land in the same clusters?). Data on a continuum gives weak silhouette no matter what k you choose.
- *Simple:* Any pile of sand can be cut into scoops. Ask whether there were separate piles.
- *Hook:* **k-means will always answer; the null tells you whether to believe it.**

**Causal inference from observational data**
- *Real:* A naive treated-vs-control gap mixes the treatment effect with confounding (here, second nights of back-to-backs are more often road games). Adjustment (OLS), weighting (IPW), or both (AIPW, doubly robust) can remove measured confounding, but never unmeasured confounding. A placebo (a "treatment" that cannot cause the outcome) tests the design. The E-value says how strong an unmeasured confounder would have to be to explain the result away.
- *Simple:* Compare like with like, then try to fool yourself and see whether you can.
- *Hook:* **Balance it, placebo it, E-value it.**

**Attention and the transformer**
- *Real:* Each position makes a query, key and value. Attention weights are softmax(QK^T / sqrt(d)); the output is a weighted sum of values. A learned [CLS] token collects a summary. Positional embeddings are needed because attention alone ignores order.
- *Simple:* Every game in the history asks "which other games matter for reading me?" and mixes information accordingly.
- *Hook:* **Query asks, key answers, value is what you get.**

**Walk-forward evaluation**
- *Real:* One train/test split gives one noisy estimate. Retraining each season and testing on the next gives many chronological out-of-sample estimates, which pool into tighter intervals and expose whether performance is stable across seasons.
- *Simple:* Don't judge a forecaster on one game; judge them every season, using only what they knew then.
- *Hook:* **Retrain, roll forward, repeat.**

## Teach-back questions (answer without looking at code)

1. In `ex5`, which exact line leaks, and what would a model trained on that feature learn to do? (Do this before you fix it.)
2. Why would a leakage test that only rewrites games after a cutoff day miss a feature that peeks at its own game's result? What did this repo do about that?
3. Can temperature scaling change accuracy? Why or why not? When does it change log loss but not accuracy?
4. A two-feature logistic regression recovered about 70% of the gain of the whole ensemble over raw Elo. What does that tell you about where the gain came from, and what should you claim on a resume?
5. Why does weighting each row by 1/propensity remove confounding? What breaks when some propensities are near 0 or 1, and what did this repo do about it?
6. What does the placebo test check? What would a large placebo effect mean for the back-to-back estimate? The actual placebo was slightly negative with an interval including zero: what do you conclude and what do you not conclude?
7. The E-value for the win effect is about 1.5. In plain English, what does that number say, and is it reassuring or not?
8. Silhouette was about 0.11 against a shuffled-data null of 0.08, yet cluster stability (ARI) was 0.78. Reconcile these. Would you call the clusters "archetypes"? Why z-score within each season before clustering?
9. In the transformer, why prepend a [CLS] token and add positional embeddings? What changes if you remove the positional embeddings? (There is a test in `tests/test_models.py` that bears on this.)
10. What does walk-forward evaluation estimate that a single split does not? Why are its confidence intervals narrower here?
11. The league's 3-point attempt rate went from about 22% (2011) to about 40% (2024). What does that imply for training a model on every season equally, and what would you try?
12. Which single result in this repo are you least confident in, and what experiment would change your mind?

## How the pieces fit together (draw this from memory)

```
box scores ──► per-game style metrics ──► trailing-20 (shift 1) ──► PCA (fit on train only) ─┐
results ─────► Elo (pre-game) ───────────► form / rest / home ───────────────────────────────┼─► models
                                                                                            │   logreg, boosting, MLP, GRU, transformer
results ─────► causal analysis (back-to-back effect) ─────► tells us the rest features matter ┘
```
