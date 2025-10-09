# line-predictions

NFL rushing and receiving yards predictions using statistical modeling (lognormal distributions) with opponent defense adjustments.

## Overview

This tool predicts player performance by:
1. Fitting lognormal distributions to historical weekly yard data
2. Adjusting predictions based on opponent defensive strength
3. Generating probabilistic predictions (median, expected, percentiles)

**Positions supported:**
- **RB** - Rushing yards
- **WR** - Receiving yards

## Setup

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd line-predictions
```

2. Navigate to the package directory:
```bash
cd line_predictions
```

3. Create virtual environment and install dependencies:
```bash
uv venv
uv sync
```

The package will be installed automatically with all dependencies.

## Usage

All commands should be run from the `line_predictions/` directory.

### Complete Workflow

To generate predictions for a specific week, run these commands in order:

#### 1. Fetch Data
```bash
uv run line-predictions fetch --season 2025 --season-type REG
```

This fetches:
- Weekly RB rushing yards
- Weekly WR receiving yards
- Team rush defense allowed
- Team pass defense allowed

#### 2. Fit Player Models (RBs)
```bash
uv run line-predictions fit-players --season 2025 --season-type REG --position-filter RB
```

#### 3. Fit Player Models (WRs)
```bash
uv run line-predictions fit-players --season 2025 --season-type REG --position-filter WR
```

#### 4. Calculate Defense Adjustments (Rush)
```bash
uv run line-predictions defense-adjustments --season 2025 --season-type REG --position RB
```

#### 5. Calculate Defense Adjustments (Pass)
```bash
uv run line-predictions defense-adjustments --season 2025 --season-type REG --position WR
```

#### 6. Generate Predictions
```bash
uv run line-predictions schedule-predictions --season 2025 --season-type REG --week 6 --top-n 30
```

This creates two separate CSV files:
- `reports/predictions_RB_2025_REG_week6.csv` - Top 30 RB rushing yard predictions
- `reports/predictions_WR_2025_REG_week6.csv` - Top 30 WR receiving yard predictions

### Output Format

Each prediction CSV contains:
- `game_id` - Unique game identifier
- `season` - Season year
- `week` - Week number
- `position` - Player position (RB or WR)
- `player_id` - Player GSIS ID
- `player_name` - Player full name
- `team` - Player's team
- `opponent` - Opponent team
- `predicted_median` - 50th percentile (fair betting line)
- `predicted_expected` - Mean expected yards
- `predicted_p75` - 75th percentile

### Additional Commands

#### Predict Single Player
```bash
uv run line-predictions predict \
  --season 2025 \
  --season-type REG \
  --position RB \
  --player-name "Derrick Henry" \
  --opponent-team "KC" \
  --line 75.5
```

Returns probability of going over the specified line.

#### Plot Player Distribution
```bash
uv run line-predictions plot \
  --season 2025 \
  --season-type REG \
  --position RB \
  --player-name "Derrick Henry" \
  --opponent-team "KC" \
  --line 75.5
```

Saves a PNG visualization to `reports/plots/`.

## Data Sources

- **nflreadpy** - Play-by-play data and rosters
- **Pro Football Reference** - Historical team and player data (fallback)

## Directory Structure

```
line-predictions/
├── line_predictions/           # Package directory
│   ├── src/
│   │   └── line_predictions/
│   │       ├── __init__.py
│   │       └── cli.py         # Main CLI implementation
│   ├── pyproject.toml         # Package configuration
│   └── uv.lock                # Dependency lock file
├── data/
│   ├── raw/                   # Fetched data (parquet files)
│   └── processed/             # Fitted models (JSON files)
├── reports/
│   ├── predictions_RB_*.csv   # RB predictions
│   ├── predictions_WR_*.csv   # WR predictions
│   └── plots/                 # Visualization outputs
└── README.md
```

## Statistical Model

The tool uses **lognormal distributions** to model player performance:

1. **Why lognormal?**
   - Right-skewed (captures explosive games)
   - No negative values
   - Multiplicative effects (good for adjustments)

2. **Opponent Adjustments:**
   - Calculates team defensive strength vs league average
   - Applies multiplicative adjustment in log-space
   - Uses cumulative season-to-date data

3. **Zero Handling:**
   - DNP (Did Not Play) or zero-carry games = 0 yards
   - Only positive yards used for distribution fitting

## Examples

### Week 6 Top Predictions

**RBs (Rushing Yards):**
1. Josh Jacobs (GB) - 131.9 median vs CIN
2. Quinshon Judkins (CLE) - 116.1 median vs PIT
3. Bijan Robinson (ATL) - 95.4 median vs BUF

**WRs (Receiving Yards):**
1. Puka Nacua (LA) - 172.0 median vs BAL
2. Jaxon Smith-Njigba (SEA) - 104.9 median vs JAX
3. Zay Flowers (BAL) - 67.5 median vs LA

## Troubleshooting

### Module Not Found Error
If you get `ModuleNotFoundError: No module named 'line_predictions'`:
```bash
cd line_predictions
rm -rf .venv
uv venv
uv sync
```

### Data Fetch Issues
Ensure you have internet connection and try with explicit season:
```bash
uv run line-predictions fetch --season 2025 --season-type REG
```

## Author

Jake Moses (jake@pedestal.ai)

## License

See LICENSE file for details.
