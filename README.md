# line-predictions

NFL player performance predictions using advanced statistical modeling (lognormal distributions) with recency weighting, usage filtering, opponent adjustments, and injury risk detection.

## Overview

Predict RB rushing yards and WR receiving yards with a sophisticated algorithm that considers:
- **Recency Weighting** - Recent games weighted exponentially higher
- **Usage Filtering** - Only reliable players with sufficient snap share
- **Opponent Adjustments** - Defensive strength vs league average
- **Injury Detection** - Flags players with unusual usage drops
- **Position-Specific** - Separate tuning for RB vs WR variance

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

### 🚀 One Command - Generate Predictions

Run everything with a single command:

```bash
uv run line-predictions predict 9
```

That's it! This command automatically:
1. ✅ Fetches latest data from nflreadpy
2. ✅ Fits RB & WR models with advanced algorithm
3. ✅ Calculates defense adjustments
4. ✅ Generates predictions with injury risk warnings

**Options:**
```bash
uv run line-predictions predict 9 --season 2025 --season-type REG --top-n 30
```

**Output:** Two CSV files in `reports/`:
- `predictions_RB_2025_REG_week9.csv` - Top 30 RB predictions
- `predictions_WR_2025_REG_week9.csv` - Top 30 WR predictions

### 📊 Output Format

Each prediction includes:

| Column | Description | Use Case |
|--------|-------------|----------|
| `player_name` | Player full name | |
| `team` | Player's team | |
| `opponent` | Opponent team | |
| `predicted_p25` | 25th percentile | Lower bound of middle 50% |
| `predicted_median` | 50th percentile | Fair betting line (50/50) |
| `predicted_p75` | 75th percentile | Upper bound of middle 50% |
| `predicted_expected` | Mean (E[X]) | Expected value |
| `injury_risk` | Risk level | none / low / medium / high |
| `injury_note` | Risk explanation | Usage drop details |

### 🎯 Betting Strategy: Middle 50% Range

The **p25-p75 range** represents a 50% confidence interval:

- **Line < p25** → **HIGH CONFIDENCE OVER** ✅ - Only 25% chance of going under
- **Line > p75** → **HIGH CONFIDENCE UNDER** ✅ - Only 25% chance of going over
- **p25 ≤ Line ≤ p75** → **Lower confidence** ⚠️ - Close to 50/50

**Example:**
```
Player: Josh Jacobs
p25: 62.3 yards
Median: 85.7 yards
p75: 115.4 yards

Betting Line: 55.5 yards → HIGH CONFIDENCE OVER (line < p25)
Betting Line: 125.5 yards → HIGH CONFIDENCE UNDER (line > p75)
Betting Line: 80.5 yards → Lower confidence (inside range)
```

### ⚠️ Injury Risk Levels

- **high** - Very low usage last week (possible injury/inactive)
- **medium** - Significant drop in snap share from season average
- **low** - Below average snap share
- **none** - Normal usage

## Advanced Algorithm Features

### 1. Recency Weighting
- Exponential decay favors recent games
- **RBs:** 0.90 decay (~3-4 week half-life)
- **WRs:** 0.85 decay (~2-3 week half-life for faster role changes)
- Recent performance weighted much higher than season averages

### 2. Usage Filtering
- Automatically filters unreliable players
- **RBs:** Minimum 30% snap share, 8 touches/game
- **WRs:** Minimum 40% snap share OR 10% target share
- Only includes players with 3+ games played

### 3. Opponent Adjustments
- Season-to-date cumulative defensive strength vs league average
- Multiplicative adjustment in log-space: `adj_mu = mu + delta_log`
- Delta = `ln(league_avg / team_avg)` yards allowed
- Tougher defenses → lower predictions, weaker defenses → higher predictions

### 4. Injury Detection
- Monitors snap percentage and touch/target trends
- Flags significant usage drops from season average
- Identifies players with very low recent usage
- **HIGH risk** = Possible injury or inactive status

### 5. Position-Specific Tuning
- **RBs:** Tighter variance (0.90x sigma) - consistent workload
- **WRs:** Wider variance (1.15x sigma) - volatile game-to-game
- Different recency decay and usage thresholds by position

## Statistical Model

Uses **lognormal distributions** because:
- Right-skewed (captures explosive games)
- No negative values (yards ≥ 0)
- Multiplicative effects work naturally in log-space
- Realistic for sports performance data

**Key Parameters:**
- `mu` - Log-space mean
- `sigma` - Log-space standard deviation (adjusted by position)
- Percentiles calculated with calibrated sigma for proper coverage

## Performance Metrics

### Running Backs
- **MAE:** 25.6 yards (34% improvement vs baseline)
- **Correlation:** 0.40 (132% improvement)
- **Directional Accuracy:** 75%
- **Coverage (50-75th):** ~25%
- **Grade:** A-

### Wide Receivers
- **MAE:** 28.2 yards (36% improvement vs baseline)
- **Correlation:** 0.37 (from negative to positive)
- **Directional Accuracy:** 63%
- **Coverage (50-75th):** ~25%
- **Grade:** B+

See `reports/IMPROVEMENT_SUMMARY.md` for detailed analysis.

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

## Data Sources

- **nflreadpy** - Play-by-play data, schedules, and rosters
- Automatically fetches:
  - Weekly player performance (yards)
  - Usage metrics (snaps, touches, targets)
  - Team defense allowed
  - Opponent matchups

## Directory Structure

```
line-predictions/
├── line_predictions/           # Package directory
│   ├── src/
│   │   └── line_predictions/
│   │       ├── __init__.py
│   │       └── cli.py         # Main CLI (simplified)
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
Ensure you have internet connection. The tool requires active NFL season data from nflreadpy.

### Predictions Look Off
- Check the `injury_risk` column - players with high risk may be injured
- Verify you're using data from the correct week
- Look at the p25-p75 range for confidence level

## Author

Jake Moses (mosesjake32@gmail.com)

## License

See LICENSE file for details.
