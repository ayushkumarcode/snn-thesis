# SNN-ESC50 Thesis Project

## Git Rules
- NEVER use Co-Authored-By in commits
- Just `git add`, `git commit -m "message"`, `git push`
- Commits must show as ayushkumarcode only
- Repo: https://github.com/ayushkumarcode/snn-thesis

## Project Structure
- Code: `snn-esc50/` with venv at `snn-esc50/.venv/`
- SpiNNaker venv: `snn-esc50/.venv-spinnaker/` (Python 3.11)
- Results: `snn-esc50/results/`
- Paper: `snn-esc50/paper/latex/main.tex`
- Research docs: `push/`

## SpiNNaker Debugging Mindset
SpiNNaker is a WORKING system. When deployment fails, it's OUR code/config problem. Debug systematically, never conclude "can't be done."

## Key Technical Notes
- `config.py` has lazy torch import (for .venv-spinnaker compatibility)
- SpiNNaker key fix: `sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 32)`
- SpiNNaker key fix: `population.initialize(v=0.0)`
- CSF3 project at `~/scratch/snn-esc50/`, partition `gpuA`
