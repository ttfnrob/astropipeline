# AstroAgent Pipeline 🚀

> *AI agents that generate and test astrophysics hypotheses automatically*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is this?

AstroAgent Pipeline uses specialized AI agents to conduct astrophysics research automatically. It generates novel hypotheses, reviews them, designs experiments, and tracks the entire research process.

**Key features:**
- 🤖 **AI agents** for each research step
- 📚 **Literature integration** - checks novelty against current papers  
- 🔄 **Full workflow** - from idea to experiment design
- 📊 **Complete tracking** - audit trail of all decisions

## 🚀 Quick Start (5 minutes)

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

# 3. Generate your first research hypothesis!
python start.py --tags "exoplanets,atmospheres" --count 3

# 4. See what was created
python start.py --status
```

**That's it!** You now have AI-generated astrophysics hypotheses ready for research.

## How it works

The pipeline uses AI agents that work together:

1. **Hypothesis Maker** - Generates novel research ideas
2. **Reviewer** - Evaluates ideas for feasibility and impact  
3. **Experiment Designer** - Creates detailed research plans
4. **Experimenter** - Executes analyses and generates results
5. **Peer Reviewer** - Validates results (coming soon)
6. **Reporter** - Writes research papers (coming soon)

Each agent checks recent literature to ensure novelty and tracks all decisions for reproducibility.

## What you can do right now

- ✅ **Generate hypotheses** - AI creates testable research ideas
- ✅ **Automatic scoring** - Ideas rated on impact, feasibility, novelty  
- ✅ **Experiment design** - Detailed protocols created automatically
- ✅ **Project tracking** - Full audit trail and progress monitoring
- 🚧 **Paper writing** - Coming soon

## Examples

```bash
# Generate 5 exoplanet research ideas
python start.py --tags "exoplanets,transit photometry" --count 5

# Review and approve ideas
python start.py --review

# Design experiments for approved ideas  
python start.py --design

# Check overall status
python start.py --status
```

## Requirements

- Python 3.9+
- Free ADS API token (for literature search)
- OpenAI API key or Claude API key (for AI agents)

## Contributing

This is an active research project! We welcome:
- Bug reports and feature requests
- New agent implementations
- Integration with additional data sources
- Documentation improvements

## License

MIT License - feel free to use for research or commercial projects.

---

*Making astrophysics research faster and more systematic* 🌟
