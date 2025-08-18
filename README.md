# AstroAgent Pipeline 🚀

> *Autonomous AI research system that takes ideas from conception to completion*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is this?

AstroAgent Pipeline is a **goal-driven continuous research system** that uses specialized AI agents to conduct astrophysics research automatically. Unlike traditional tools that require manual intervention at each step, this system runs autonomously until research goals are achieved.

**Key features:**
- 🎯 **Goal-driven execution** - auto-pauses when research targets are completed
- 🔄 **Continuous pipeline** - ideas flow through all stages without manual intervention
- 🤖 **Smart AI agents** - specialized for each research step
- 📚 **Literature integration** - real-time novelty checking against current papers  
- 🌐 **Real-time web dashboard** - monitor progress and control pipeline
- 📊 **Complete research audit trail** - track every decision and result

## 🚀 Quick Start (2 minutes)

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/astropipeline.git
cd astropipeline
pip install -r requirements.txt

# 2. Get free API keys and add to .env file
# ADS token: https://ui.adsabs.harvard.edu/user/settings/token
# OpenAI key: https://platform.openai.com/api-keys
echo "ADS_API_TOKEN=your_ads_token_here" > .env
echo "OPENAI_API_KEY=your_openai_key_here" >> .env

# 3. Start the autonomous research system!
python start.py --domains "exoplanets,atmospheres"
```

**That's it!** 

- The system starts generating hypotheses, reviewing them, designing experiments, and executing research
- A web dashboard opens at `http://localhost:8000` to monitor real-time progress  
- The pipeline automatically **pauses when the first idea completes the full research workflow**
- Click any idea in the dashboard for detailed information
- Use the **Pause/Resume** button to control execution

**🎯 Goal-driven approach:** Instead of generating many disconnected ideas, the system focuses on advancing good ideas through the complete research pipeline until achievement criteria are met.

## How it works

The system runs a **continuous multi-agent research pipeline**:

### 🔄 Continuous Pipeline Flow
1. **Smart Hypothesis Generation** - Creates ideas only when pipeline needs more work (avoids hypothesis overload)
2. **Automatic Review** - AI evaluates ideas for novelty, feasibility, and impact using current literature
3. **Experiment Design** - Approved ideas get detailed research protocols automatically  
4. **Execution Simulation** - Research plans are validated and marked ready for implementation
5. **Completion Tracking** - Finished work is archived and counted toward goals

### 🎯 Goal-Driven Execution
- **Default behavior**: Auto-pause after **1 completed idea** (configurable)
- **Smart resource management**: Only generates new hypotheses when pipeline needs more active work  
- **Focus on quality**: Advances promising ideas through all stages before creating new ones
- **Achievement-based stopping**: Stops when research goals are met, not arbitrary time limits

### 🌐 Real-Time Dashboard
- Monitor all ideas progressing through pipeline stages
- See agent activity with live status indicators  
- Pause/Resume/Stop controls for pipeline management
- Sortable, filterable tables with detailed idea information
- Click any idea for comprehensive details modal

## What you can do right now

- ✅ **Autonomous research execution** - Full pipeline from idea to completion without manual intervention
- ✅ **Real-time monitoring** - Web dashboard shows live progress and agent activity
- ✅ **Goal-driven operation** - Automatically pauses when research targets achieved  
- ✅ **Smart idea management** - Only generates new hypotheses when pipeline needs more work
- ✅ **Interactive control** - Pause/Resume/Stop pipeline via web interface
- ✅ **Detailed idea exploration** - Click any idea for comprehensive information
- 🚧 **Actual experiment execution** - Currently simulated, real data analysis coming soon

## Usage Examples

```bash
# Default: Run continuous pipeline + web dashboard (auto-pause after 1 completed idea)
python start.py

# Run until 3 ideas are completed, then auto-pause
python start.py --complete-ideas 3

# Focus on specific research areas
python start.py --domains "stellar evolution,supernovae"

# Web dashboard only (useful if pipeline already running elsewhere)
python start.py web

# Pipeline only (no web interface)  
python start.py pipeline

# Traditional discrete mode (generate ideas once, then stop)
python start.py pipeline --mode discrete --count 5
```

### Web Dashboard Features
- **Real-time progress tracking** - Watch ideas advance through pipeline stages
- **Agent status monitoring** - See which agents are active with visual indicators
- **Pipeline controls** - Pause, Resume, or Stop execution
- **Interactive tables** - Sort and filter ideas by status, score, domain  
- **Detailed idea modals** - Click any row for full hypothesis, rationale, and research details

## Requirements

- **Python 3.9+**
- **Free ADS API token** (for literature search) - [Get yours here](https://ui.adsabs.harvard.edu/user/settings/token)
- **OpenAI API key** (for AI agents) - [Get yours here](https://platform.openai.com/api-keys)

## Advanced Configuration

```bash
# Run with custom completion targets and time limits
python start.py --complete-ideas 5 --max-time 60  # 5 ideas OR 60 minutes max

# Adjust pipeline sensitivity (how many ideas to keep active)
python start.py --min-pipeline-size 2  # Keep 2-3 active ideas flowing

# Different research domains
python start.py --domains "galactic dynamics,dark matter,stellar formation"

# Use discrete mode for traditional batch processing
python start.py pipeline --mode discrete --count 10
```

## Contributing

This is an active research project pushing the boundaries of **autonomous AI research systems**! We welcome:
- 🐛 Bug reports and feature requests
- 🤖 New agent implementations  
- 📊 Integration with real astronomical data sources
- 🔬 Novel research pipeline strategies
- 📚 Documentation improvements

## License

MIT License - feel free to use for research or commercial projects.

---

*Autonomous AI research systems for the next generation of scientific discovery* 🌟
