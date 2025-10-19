# Agent Context: Line Predictions

## Project Overview

NFL rushing (RB) and receiving (WR) yards prediction system using statistical modeling with lognormal distributions and opponent defense adjustments. Built with Python 3.13+ using the `uv` package manager.

**Primary Goal:** Generate probabilistic predictions for player performance to identify high-confidence betting opportunities.

## Quick Command Reference

### Main Command (All-in-One)
```bash
cd line_predictions
uv run line-predictions generate-predictions --season 2025 --season-type REG --week 7
```

### Individual Steps (if needed)
```bash
# 1. Fetch data
uv run line-predictions fetch --season 2025 --season-type REG

# 2. Fit player models (with advanced algorithm)
uv run line-predictions fit-players --season 2025 --season-type REG --position-filter RB --use-weighting --use-usage-filter
uv run line-predictions fit-players --season 2025 --season-type REG --position-filter WR --use-weighting --use-usage-filter

# 3. Calculate defense adjustments
uv run line-predictions defense-adjustments --season 2025 --season-type REG --position RB
uv run line-predictions defense-adjustments --season 2025 --season-type REG --position WR

# 4. Generate predictions
uv run line-predictions schedule-predictions --season 2025 --season-type REG --week 7 --top-n 30
```

### Other Useful Commands
```bash
# Backtest the model
uv run line-predictions backtest --season 2025 --season-type REG --train-weeks-str "1,2,3,4" --test-weeks-str "5,6" --position RB

# Single player prediction
uv run line-predictions predict --season 2025 --season-type REG --position RB --player-name "Derrick Henry" --opponent-team "KC" --line 75.5

# Plot player distribution
uv run line-predictions plot --season 2025 --season-type REG --position RB --player-name "Derrick Henry" --opponent-team "KC" --line 75.5
```

## Directory Structure

```
line-predictions/
├── line_predictions/           # Package directory
│   ├── src/
│   │   └── line_predictions/
│   │       ├── __init__.py
│   │       └── cli.py         # Main CLI implementation (1426 lines)
│   ├── pyproject.toml         # Package configuration
│   └── uv.lock                # Dependency lock file
├── data/
│   ├── raw/                   # Fetched data (parquet files)
│   │   ├── weekly_RB_2025_REG.parquet
│   │   ├── weekly_WR_2025_REG.parquet
│   │   ├── weekly_RB_usage_2025_REG.parquet
│   │   ├── weekly_WR_usage_2025_REG.parquet
│   │   ├── team_rush_allowed_2025_REG.parquet
│   │   └── team_pass_allowed_2025_REG.parquet
│   └── processed/             # Fitted models (JSON files)
│       ├── player_lognorm_RB_2025_REG.json
│       ├── player_lognorm_WR_2025_REG.json
│       ├── team_defense_delta_RB_2025_REG.json
│       └── team_defense_delta_WR_2025_REG.json
├── reports/                   # Predictions and analysis
│   ├── predictions_RB_2025_REG_week*.csv
│   ├── predictions_WR_2025_REG_week*.csv
│   ├── backtest_*.csv
│   └── plots/
├── README.md                  # User documentation
├── AGENTS.md                  # This file
└── validate_improvements.py   # Validation script
```

## Statistical Model

### Core Algorithm: Lognormal Distribution

**Why Lognormal?**
- Right-skewed distribution (captures explosive games)
- No negative values (yards ≥ 0)
- Multiplicative effects work well in log-space
- Realistic for sports performance data

### Advanced Algorithm Features (Enabled by Default)

The "advanced algorithm" includes these improvements:

#### 1. Recency Weighting
- Exponential decay: `w_i = decay_factor^(max_week - week_i)`
- **RB decay:** 0.90 (~3-4 week half-life) - more stable roles
- **WR decay:** 0.85 (~2-3 week half-life) - faster role changes
- Recent games weighted higher than season averages

#### 2. Usage Filtering
Filters out unreliable players based on usage metrics:

**RB Thresholds:**
- Minimum 30% snap share (recent 4 games)
- Minimum 8 touches/game (rush attempts + receptions)
- Minimum 3 games played

**WR Thresholds:**
- Minimum 40% snap share OR 10% target share
- Minimum 3 games played

#### 3. Position-Specific Tuning
**RBs:**
- Sigma adjustment: 0.90x (tighter variance, consistent workload)
- Calibration factor: 1.3x (for prediction intervals)

**WRs:**
- Sigma adjustment: 1.15x (wider variance, volatile performance)
- Calibration factor: 1.5x (larger uncertainty)

#### 4. Opponent Defense Adjustments
- Season-to-date cumulative defensive strength vs league average
- Multiplicative adjustment in log-space: `adj_mu = mu + delta_log`
- Delta = `ln(league_avg / team_avg)` yards allowed

### Model Parameters

Each player model stores:
```json
{
  "player_id": "00-0012345",
  "player_name": "Player Name",
  "mu": 3.85,                    // Log-space mean
  "sigma": 0.68,                 // Log-space std dev
  "expected": 65.2,              // E[X] = exp(mu + 0.5*sigma^2)
  "games_played": 6,
  "p10": 25.4,
  "p25": 35.8,
  "p50": 52.1,
  "p75": 75.6,
  "p90": 103.2
}
```

## Prediction Output Format

Each prediction includes:

| Column | Description | Use Case |
|--------|-------------|----------|
| `predicted_p25` | 25th percentile | Lower bound of middle 50% |
| `predicted_median` | 50th percentile | Fair betting line (50/50) |
| `predicted_p75` | 75th percentile | Upper bound of middle 50% |
| `predicted_expected` | Mean (E[X]) | Expected value (used for ranking) |

### Betting Strategy: Middle 50% Range

The **p25-p75 range** represents a 50% confidence interval:

- **Line < p25** → **OVER (high confidence)** - Only 25% chance of going under
- **Line > p75** → **UNDER (high confidence)** - Only 25% chance of going over
- **p25 ≤ Line ≤ p75** → **Toss-up** - Lower confidence, close to 50/50

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

## Performance Metrics

### Running Backs (Week 5-6 Backtest)
- **MAE:** 25.6 yards (34% improvement vs baseline)
- **Correlation:** 0.40 (132% improvement)
- **Directional Accuracy:** 75%
- **Coverage (50-75th):** ~25%
- **Grade:** A-

### Wide Receivers (Week 5-6 Backtest)
- **MAE:** 28.2 yards (36% improvement vs baseline)
- **Correlation:** 0.37 (from negative to positive)
- **Directional Accuracy:** 63%
- **Coverage (50-75th):** ~25%
- **Grade:** B+

## Key Configuration Constants

Located in `cli.py` (lines 47-64):

```python
RB_CONFIG = {
    "recency_decay": 0.90,
    "min_snap_pct": 0.30,
    "min_games": 3,
    "sigma_adjust": 0.90,
    "min_touches_per_game": 8,
    "sigma_calibration": 1.3,
}

WR_CONFIG = {
    "recency_decay": 0.85,
    "min_snap_pct": 0.40,
    "min_target_share": 0.10,
    "min_games": 3,
    "sigma_adjust": 1.15,
    "sigma_calibration": 1.5,
}
```

## Data Flow

1. **Fetch** (`fetch()` command)
   - Uses `nflreadpy` to load play-by-play data
   - Aggregates weekly yards (rushing for RB, receiving for WR)
   - Calculates usage metrics (snaps, touches, targets, shares)
   - Computes team defense allowed (rush/pass yards per game)
   - Saves to `data/raw/*.parquet`

2. **Fit Players** (`fit_players()` command)
   - Loads weekly yards data
   - Applies opponent normalization (scales by defensive strength)
   - Filters by usage thresholds (if enabled)
   - Fits lognormal distribution with recency weighting
   - Applies position-specific sigma adjustments
   - Saves parameters to `data/processed/player_lognorm_*.json`

3. **Defense Adjustments** (`defense_adjustments()` command)
   - Calculates season-to-date team averages
   - Computes delta vs league average in log-space
   - Saves to `data/processed/team_defense_delta_*.json`

4. **Generate Predictions** (`schedule_predictions()` command)
   - Loads schedules from `nflreadpy`
   - Matches players to teams and opponents
   - Adjusts player mu by opponent defense delta
   - Calculates percentiles and expected values
   - Ranks top N players by expected yards
   - Saves separate CSV files for RB and WR

## Important Functions

### Core Statistical Functions
- `_fit_lognormal_weighted()` (line 95) - Fits lognormal with recency weighting
- `_expected_from_params()` (line 138) - Calculates E[X] from mu, sigma
- `_percentiles_from_params()` (line 143) - Calculates percentiles
- `_percentiles_from_params_calibrated()` (line 148) - Calibrated percentiles

### CLI Commands
- `fetch()` (line 302) - Fetch all raw data
- `fit_players()` (line 514) - Fit player models
- `defense_adjustments()` (line 714) - Calculate defense deltas
- `predict()` (line 757) - Single player prediction
- `plot()` (line 822) - Visualize distribution
- `schedule_predictions()` (line 964) - Generate full predictions
- `backtest()` (line 1133) - Backtest model performance
- `generate_predictions()` (line 1347) - **NEW: All-in-one pipeline command**

## Recent Changes (October 19, 2025)

### Commit: "feat: Add pipeline command and middle 50% range"

**What was added:**
1. **Single Pipeline Command** (`generate-predictions`)
   - Runs entire workflow in one command
   - Eliminates 6-step manual process
   - Shows progress for each step
   - Includes helpful betting tips in output

2. **Middle 50% Confidence Range**
   - Added `predicted_p25` to all predictions
   - Shows where 50% of outcomes likely fall
   - Identifies high-confidence betting opportunities
   - Column order: p25 → median → p75 → expected

3. **Updated Documentation**
   - Quick Start section in README
   - Betting strategy guide for p25-p75 range
   - Clear explanation of confidence intervals

## Data Sources

- **Primary:** `nflreadpy` - Play-by-play data, schedules, rosters
- **Fallback:** Pro Football Reference (PFR) via web scraping (not currently used but functions exist)

## Common Troubleshooting

### "ModuleNotFoundError: No module named 'line_predictions'"
```bash
cd line_predictions
rm -rf .venv
uv venv
uv sync
```

### Data fetch issues
- Ensure internet connection
- `nflreadpy` requires active NFL season data
- May need to wait for data availability after games

### Missing predictions for a player
- Check usage thresholds - player may be filtered out
- Verify player played in training weeks
- Check position filter (RB vs WR)

## Developer Notes

### Testing Changes
- Use `backtest` command to evaluate model performance
- Compare MAE, correlation, directional accuracy
- Coverage should be ~25% for p50-p75 interval

### Adding New Features
- Position configs are in `RB_CONFIG` and `WR_CONFIG` dictionaries
- Most commands use Typer for CLI arguments
- Rich console for formatted output
- All paths relative to `DATA_DIR`, `REPORTS_DIR`, `PLOTS_DIR`

### Code Style
- Type hints used throughout
- Pandas for data manipulation
- NumPy/SciPy for statistics
- Pathlib for file paths

## Links & Resources

- **Repository:** (Add GitHub URL here)
- **Author:** Jake Moses (mosesjake32@gmail.com)
- **Documentation:** See README.md
- **Performance Analysis:** See reports/IMPROVEMENT_SUMMARY.md

## Future Enhancements to Consider

- Add weather data adjustments
- Incorporate player injury status
- Home/away field advantages
- Divisional game adjustments
- Over/under market line integration
- Real-time odds comparison
- Multi-week projection trends
- Ensemble methods (combining multiple models)

