# NEXUS

NEXUS is an experimental cognitive architecture exploring general-purpose, experience-driven intelligence through explicit memory, world modelling, prediction, action selection, and evaluation.

**NEXUS is not AGI, and this repository does not claim that AGI has been achieved.** The goal is to build and rigorously evaluate increasingly capable cognitive mechanisms. This first milestone is a small, inspectable research prototype.

## First milestone

- SQLite-backed episodic memory.
- Explicit beliefs with confidence and evidence provenance.
- An action-conditioned model that learns average numeric state changes from experience.
- Prediction and prediction-error tracking.
- Goal-directed selection among actions explicitly supplied by an environment.
- A cycle of observe → predict → choose → act → evaluate → learn.
- CLI sandbox, tests, and GitHub Actions.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
pytest
nexus
```

Try these commands in the NEXUS prompt:

```text
observe energy=1 knowledge=0
goal energy=10
actions recharge,wait,explore
step
status
quit
```

## Architecture and limitations

The current world model is deliberately simple: it averages numeric state deltas observed after named actions. The selector estimates which supplied action moves the current state toward a numeric goal. It does not yet represent rich causal structure, language understanding, perception, long-horizon planning, or general learning.

The cognitive core has no network, shell, or arbitrary tool permissions. Environments supply the available actions and execute them. Future integrations must use scoped capabilities, audit logs, and human approval for consequential operations.

## Roadmap

- [x] Episodic memory
- [x] Explicit belief records with evidence
- [x] Experience-conditioned transition learning
- [x] Prediction error and action scoring
- [ ] Relational and causal world models
- [ ] Hierarchical planning and subgoals
- [ ] Active experiment design and information gain
- [ ] Continual-learning benchmarks and forgetting tests
- [ ] Richer perception and environment interfaces
- [ ] Uncertainty calibration and out-of-distribution evaluation
