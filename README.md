# MARL Soccer Self-Play
A lightweight 2D football environment for multi-agent reinforcement learning with optional rendering, reward shaping, and self-play training tools.

## Features
- Top-down 2D pitch with continuous actions and partial observations per agent.
- Works from 1v1 up to 11v11; naive scripted bots included for quick demos.
- Reward shaping hooks (ball progress, proximity, possession) to speed up learning.
- Pygame renderer for visual debugging; headless runs by disabling render.

## Setup
### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart Demos
- 11v11 scripted vs scripted (renders by default):
  ```powershell
  python main.py
  ```
- Lightweight 1v1 demo (optionally load trained weights):
  ```powershell
  python run_1v1_demo.py --game-length 1200 --policy-weights path/to/weights.weights.h5
  ```
  If no weights are given, both teams use the naive attention bot.

## Training With Self-Play
Train a shared actor-critic against a snapshot pool of past policies.
```powershell
python train_agents.py --episodes 200 --game-length 1200 --num-per-team 1 --log-path train_metrics.csv \
  --pool-size 6 --pool-latest-prob 0.6 --add-every 50 --add-threshold -1 \
  --dense-shaping-coef 0.02 --proximity-coef 0.01 --possession-coef 0.005
```
Key flags:
- `--num-per-team`: players per side (1–11).
- `--render`: open a Pygame window (slower; disable for headless training).
- `--save-weights`: where to save learned policy (auto-suffix `.weights.h5`).
- `--dense-shaping-coef`, `--proximity-coef`, `--possession-coef`: per-step shaping bonuses/penalties.
- `--pool-size`, `--pool-latest-prob`, `--add-every`, `--add-threshold`: self-play opponent pool settings.

## Plot Metrics
Convert the training CSV to a PNG curve.
```powershell
python plot_training.py --input train_metrics.csv --output training_curve.png --smooth 20 --show-lengths
```

## Repository Map
- `custom_football_env.py`: core environment, physics, rendering, observations, reward shaping hooks.
- `env_wrapper.py`: dm_env-style wrapper with batching, shaping coefficients, and conversion helpers.
- `fixed_agent.py`: naive scripted bots (random, attention, team-focused behaviors).
- `train_agents.py`: actor-critic training loop with self-play pool and CSV logging.
- `run_1v1_demo.py`: minimal 1v1 demo with shims for missing dm_env/acme deps.
- `main.py`: default 11v11 scripted vs scripted loop (renders).
- `plot_training.py`: read CSV metrics and save a plot.
- `Images/`, `weights/`: sample visuals and stored weights/plots (if generated).

## Notes
- Rendering: `include_wait=True` slows steps for viewing; set `render=False` (and keep `include_wait=False`) for faster training.
- Requirements target TensorFlow 2.8 and pygame 2.1.2; newer versions may work but are untested.
