# Hypothesis Lab

Hypothesis Lab is a small experimental framework for studying how
different agents actively select scientific experiments to distinguish
between competing hypotheses.

The project uses a simulated pendulum as a controlled environment.
Agents observe experimental measurements, fit candidate physical models,
select follow-up experiments, and update their beliefs from the resulting
evidence.

The goal is not to study pendulum physics itself, but to explore simple
algorithms for **active scientific experimentation and hypothesis
selection**.

---

## Overview

The hidden environment follows the simple pendulum relationship:

$T = 2\pi\sqrt{\frac{L}{g}}$ 


where:

- \(T\) is the oscillation period,
- \(L\) is pendulum length,
- \(g\) is gravitational acceleration.

The agents do not receive this equation directly.

Instead, they must distinguish between four candidate hypotheses:

1. Constant period
2. Linear dependence on length
3. Square-root dependence on length
4. Square-root dependence on length and gravity

The initial observations intentionally keep gravity fixed, which makes
the two square-root hypotheses initially difficult to distinguish.

Agents must therefore decide which experiment should be performed next.

---

## Agents

Four experiment-selection strategies are implemented.

### Random Agent

Randomly selects an available experiment.

This provides a baseline for determining whether active experiment
selection improves discovery efficiency.

### Greedy Agent

Ranks the current hypotheses and compares the two best candidates.

It selects the experiment where their predictions differ the most.

### Disagreement Agent

Considers predictions from all current hypotheses.

For every candidate experiment, it calculates the average pairwise
disagreement between predictions and selects the experiment with the
largest disagreement.

### Information Gain Agent

Maintains relative weights over competing hypotheses and estimates the
entropy of the current hypothesis distribution.

For each candidate experiment, it estimates the expected posterior
entropy and selects the experiment with the largest expected information
gain:

$ IG(e) = H(H \mid D) - \mathbb{E}[H(H \mid D, e, y)]$

where:

- \(H\) represents the hypothesis space,
- \(D\) represents the observations collected so far,
- \(e\) is a candidate experiment,
- \(y\) is its possible outcome.

---

## Discovery Loop

Each discovery iteration follows the same general process:

```text
Observations
     ↓
Fit competing hypotheses
     ↓
Rank / weight hypotheses
     ↓
Select next experiment
     ↓
Execute experiment
     ↓
Collect new observation
     ↓
Update hypotheses
     ↓
Repeat or stop

```markdown
```

The loop stops when the system reaches sufficient confidence in one
hypothesis or exhausts its experimental budget.

---

## Observation Noise

The pendulum environment supports Gaussian measurement noise:

$y = T + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma^2)$

The benchmark evaluates five noise conditions:

* $\sigma = 0.00$
* $\sigma = 0.10$
* $\sigma = 0.25$
* $\sigma = 0.50$
* $\sigma = 1.00$

Each agent/noise combination is evaluated across **100 random seeds**
with a maximum budget of **10 additional experiments**.

---

## Results

The following results were obtained from the current benchmark.

Each cell reports:

**success rate / mean number of experiments**

| Noise $\sigma$ |      Random |      Greedy | Disagreement | Information Gain |
| -------------: | ----------: | ----------: | -----------: | ---------------: |
|           0.00 | 100% / 2.08 | 100% / 1.00 |  100% / 3.00 |      100% / 1.00 |
|           0.10 | 100% / 3.50 | 100% / 1.02 |   99% / 3.03 |      100% / 1.02 |
|           0.25 |  90% / 7.64 |  93% / 4.15 |   91% / 5.84 |       93% / 4.23 |
|           0.50 |  75% / 9.97 |  74% / 9.35 |   73% / 9.36 |       79% / 9.44 |
|           1.00 | 60% / 10.00 |  61% / 9.89 |   54% / 9.83 |       55% / 9.86 |

At low noise levels, targeted experiment selection substantially reduces
the number of experiments required compared with random selection.

As observation noise increases, the distinction between hypotheses
becomes less reliable and performance decreases across all strategies.

The results also illustrate that more complex experiment-selection
strategies are not automatically superior. Their effectiveness depends
on the hypothesis space, uncertainty assumptions, available experiments,
and observation noise.

---

## Reproducibility

The benchmark stores both aggregate and per-run results.

```text
results/
├── benchmark_summary.csv
└── benchmark_runs.csv
```

### `benchmark_summary.csv`

Contains one row per agent and noise condition, including:

* success rate
* mean number of experiments
* median number of experiments
* minimum number of experiments
* maximum number of experiments

### `benchmark_runs.csv`

Contains every individual benchmark run, including:

* agent
* noise level
* random seed
* success or failure
* number of experiments
* final selected hypothesis
* fitted parameter
* final MSE
* sequence of selected experiments

This makes all reported aggregate results traceable to individual runs.

---

## Project Structure

```text
hypothesis-lab/
├── results/
│   ├── benchmark_summary.csv
│   └── benchmark_runs.csv
│
├── scripts/
│   ├── benchmark_agents.py
│   └── run_pendulum.py
│
├── src/
│   └── hypothesis_lab/
│       ├── agents/
│       │   ├── base.py
│       │   ├── random_agent.py
│       │   ├── greedy_agent.py
│       │   ├── disagreement_agent.py
│       │   └── information_gain_agent.py
│       │
│       ├── environments/
│       │   └── pendulum.py
│       │
│       ├── experiments/
│       │   ├── experiment.py
│       │   └── observation.py
│       │
│       ├── hypotheses/
│       │   ├── hypothesis.py
│       │   ├── posterior.py
│       │   └── scorer.py
│       │
│       └── runner/
│           └── experiment_loop.py
│
└── README.md
```

---

## Running the Demo

Run a single pendulum discovery example:

```bash
python scripts/run_pendulum.py
```

---

## Running the Benchmark

Run the full agent benchmark:

```bash
python scripts/benchmark_agents.py
```

The benchmark automatically writes its outputs to the `results/`
directory.

---

## What This Project Explores

This project is a small controlled study of several ideas used in
automated scientific reasoning:

* hypothesis representation
* model fitting
* active experiment selection
* uncertainty over competing hypotheses
* information gain
* noisy scientific observations
* stopping criteria
* reproducible benchmarking

It serves as a simple foundation for more complex work involving
language models, natural language processing, scientific literature,
and agentic scientific reasoning.

---

## Limitations

This project intentionally uses a small and predefined hypothesis space.

The agents do not generate new physical theories, interact with
scientific literature, or reason over natural language. The pendulum
environment is also a simplified simulator rather than a real scientific
dataset.

These constraints make the environment useful as a controlled testbed
for experiment-selection algorithms, but the framework is not intended
to represent full autonomous scientific discovery.

