# AstroAgent Pipeline 🚀

> *Autonomous AI research system that takes ideas from conception to completion*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is this?

AstroAgent Pipeline is a **goal-driven continuous research system** that uses specialized AI agents to conduct astrophysics research automatically. It runs autonomously until research goals are achieved, with a live web dashboard for monitoring and control.

**Key features:**
- 🎯 **Goal-driven execution** – auto-pauses when research targets are completed
- 🔄 **Continuous pipeline** – ideas flow through all stages without manual intervention
- 🤖 **Specialized AI agents** – tailored to each research step
- 📚 **Literature integration** – novelty checks against current papers
- 🌐 **Real-time web dashboard** – monitor and control the pipeline
- 📊 **Complete audit trail** – track every decision and result

## 🚀 Quick Start (2 minutes)

```bash
# 1) Clone and install
git clone https://github.com/yourusername/astroagent.git
cd astroagent
pip install -r requirements.txt

# 2) Create a .env (template provided) and add keys
cp environment_template.txt .env
# Required: ADS_API_TOKEN, OPENAI_API_KEY
# Optional: ANTHROPIC_API_KEY, NASA_API_KEY, etc.

# 3) Start the autonomous research system (pipeline + web UI)
python start.py --domains "exoplanets,atmospheres"
```

Notes:
- On first run, the system will create data directories automatically.
- The web dashboard is served at `http://localhost:8000`.
- By default, the pipeline **auto-pauses after the first completed idea** (configurable).

## How it works

The system runs a **continuous multi-agent research pipeline**:

### 🔄 Continuous Pipeline Flow
1. **Hypothesis Generation** – Creates ideas only when the pipeline needs more work
2. **Automatic Review** – Evaluates novelty, feasibility, and impact using literature
3. **Experiment Design** – Produces detailed protocols for approved ideas  
4. **Execution Simulation** – Validates research plans and marks them ready
5. **Completion Tracking** – Archives finished work and updates metrics

### 🎯 Goal-Driven Execution
- Default: auto-pause after **1 completed idea**
- Generates new hypotheses only when needed
- Focuses resources on advancing promising ideas end-to-end

### 🌐 Real-Time Dashboard
- Live status of agents and pipeline stages
- Pause/Resume/Stop controls
- Sortable/filterable tables and detailed idea views

## Usage

```bash
# Default: run pipeline + web UI together (auto-pauses after 1 completed idea)
python start.py

# Run until 3 ideas are completed (then auto-pause)
python start.py --complete-ideas 3

# Focus on specific research areas
python start.py --domains "stellar evolution,supernovae"

# Web UI only / Pipeline only
python start.py web
python start.py pipeline

# Discrete mode (generate once and stop)
python start.py pipeline --mode discrete --count 5

# Time-limit safeguard (continuous mode)
python start.py pipeline --complete-ideas 3 --max-time 60

# Demo and quick test
python start.py demo
python start.py test

# Reset data/projects (with confirmation)
python start.py clean
python start.py --fresh --force pipeline
```

CLI flags (subset):
- `--domains` comma-separated research domains
- `--mode` `continuous` (default) or `discrete`
- `--count` number of hypotheses for generation (default: 3)
- `--complete-ideas` target number to complete (default: 1)
- `--max-time` runtime limit in minutes
- `--fresh` reset registries and project folders; use `--force` to skip prompt
- `--skip-checks` skip env/dependency checks

## Makefile shortcuts

```bash
make setup                 # venv + install dev deps
make test                  # run tests with coverage
make lint                  # flake8 + mypy
make format                # black + isort

# Agent pipeline ops
make hypotheses TAGS="stellar dynamics" N=10
make review
make plan IDEA=01J...
make execute IDEA=01J...
make peerreview IDEA=01J...
make report IDEA=01J...
```

## Environment

Minimum requirements:
- **Python 3.9+**
- **ADS_API_TOKEN** – get from `https://ui.adsabs.harvard.edu/user/settings/token`
- **OPENAI_API_KEY** – get from `https://platform.openai.com/api-keys`

Optional keys supported (see `environment_template.txt`): `ANTHROPIC_API_KEY`, `NASA_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `MAST_TOKEN`, `SERPAPI_KEY`, `TAVILY_API_KEY`, etc.

## Docker (optional)

```bash
make docker-build
make docker-run
```

## Contributing

This is an active research project pushing the boundaries of **autonomous AI research systems**. We welcome:
- 🐛 Bug reports and feature requests
- 🤖 New agent implementations  
- 📊 Integrations with real astronomical data sources
- 🔬 Novel research pipeline strategies
- 📚 Documentation improvements

## License

MIT License – free for research and commercial use.

---

*Autonomous AI research systems for the next generation of scientific discovery* 🌟
