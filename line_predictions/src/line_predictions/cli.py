from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from scipy.stats import lognorm

import nfl_data_py as nfl
import nflreadpy as nr
import re
import requests
from bs4 import BeautifulSoup


app = typer.Typer(add_completion=False, help="NFL rushing yard predictions for RBs (2025).")
console = Console()


DATA_DIR = Path(os.getcwd()).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path(os.getcwd()).parent / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

PFR_BASE = "https://www.pro-football-reference.com"

# Map PFR team codes to common 2-3 letter NFL codes
PFR_TO_STD: Dict[str, str] = {
    "crd": "ARI", "atl": "ATL", "rav": "BAL", "buf": "BUF", "car": "CAR",
    "chi": "CHI", "cin": "CIN", "cle": "CLE", "dal": "DAL", "den": "DEN",
    "det": "DET", "gnb": "GB",  "htx": "HOU", "clt": "IND", "jax": "JAX",
    "kan": "KC",  "rai": "LV",  "lac": "LAC", "sdg": "LAC", "ram": "LAR",
    "mia": "MIA", "min": "MIN", "nwe": "NE",  "nor": "NO",  "nyg": "NYG",
    "nyj": "NYJ", "phi": "PHI", "pit": "PIT", "sfo": "SF",  "sea": "SEA",
    "tam": "TB",  "oti": "TEN", "was": "WAS",
}

PFR_TEAMS: List[str] = list(PFR_TO_STD.keys())

# Position-specific configuration
RB_CONFIG = {
    "recency_decay": 0.90,           # Exponential decay for recency weighting
    "min_snap_pct": 0.30,            # Minimum 30% snap share in recent games
    "min_games": 3,                  # Minimum games played
    "sigma_adjust": 0.90,            # Reduce variance (tighter distribution)
    "min_touches_per_game": 8,       # Minimum touches (rush + receptions)
    "sigma_calibration": 1.3,        # Inflate sigma for better coverage
}

WR_CONFIG = {
    "recency_decay": 0.85,           # Faster decay for WRs (roles change quicker)
    "min_snap_pct": 0.40,            # Minimum 40% snap share
    "min_target_share": 0.10,        # Or 10% target share
    "min_games": 3,                  # Minimum games played
    "sigma_adjust": 1.15,            # Increase variance (wider distribution)
    "sigma_calibration": 1.5,        # Larger inflation for WRs
}


def _ensure_dirs() -> None:
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _zero_as_zero_log(values: pd.Series) -> pd.Series:
    # Count negatives as 0, DNP/NaN as 0
    cleaned = values.fillna(0).clip(lower=0)
    # log(0) defined as 0 per user request
    return cleaned.apply(lambda y: np.log(y) if y > 0 else 0.0)


def _fit_lognormal_from_yards(yards: pd.Series) -> tuple[float, float]:
    """Fit lognormal parameters using only positive rushing yards.
    Zeros are excluded from the fit (lognormal supports y>0 only).
    """
    eps = 1e-8
    x = yards.fillna(0).clip(lower=0)
    pos = x[x > 0]
    if pos.empty:
        # Degenerate: no positive yards; return near-zero distribution
        return float(np.log(eps)), float(eps)
    shape, loc, scale = lognorm.fit(pos, floc=0)
    mu = float(np.log(scale))
    sigma = float(shape)
    return mu, sigma


def _fit_lognormal_weighted(yards: pd.Series, weeks: pd.Series, decay_factor: float = 0.9) -> tuple[float, float]:
    """Fit lognormal parameters with exponential recency weighting.
    
    More recent weeks get higher weight: w_i = decay_factor^(max_week - week_i)
    
    Args:
        yards: Yards gained (positive values only used for fitting)
        weeks: Week numbers corresponding to each yards value
        decay_factor: Exponential decay factor (0 < decay_factor <= 1)
                     0.9 = ~3-4 week half-life, 0.85 = ~2-3 week half-life
    
    Returns:
        (mu, sigma) parameters for lognormal distribution
    """
    eps = 1e-8
    
    # Filter to positive yards only
    mask = (yards > 0) & (yards.notna())
    if not mask.any():
        return float(np.log(eps)), float(eps)
    
    pos_yards = yards[mask].values
    pos_weeks = weeks[mask].values
    
    # Calculate recency weights
    max_week = pos_weeks.max()
    weights = np.power(decay_factor, max_week - pos_weeks)
    weights = weights / weights.sum()  # Normalize
    
    # Weighted mean and std in log-space
    log_yards = np.log(pos_yards)
    mu_weighted = float(np.sum(weights * log_yards))
    
    # Weighted variance
    var_weighted = float(np.sum(weights * (log_yards - mu_weighted) ** 2))
    sigma_weighted = float(np.sqrt(var_weighted))
    
    # Ensure minimum sigma for numerical stability
    sigma_weighted = max(sigma_weighted, eps)
    
    return mu_weighted, sigma_weighted


def _expected_from_params(mu: float, sigma: float) -> float:
    # E[X] for lognormal with parameters mu, sigma
    return float(np.exp(mu + 0.5 * sigma * sigma))


def _percentiles_from_params(mu: float, sigma: float, ps: List[float]) -> dict[str, float]:
    scale = float(np.exp(mu))
    return {f"p{int(p*100)}": float(lognorm.ppf(p, sigma, scale=scale)) for p in ps}


def _percentiles_from_params_calibrated(mu: float, sigma: float, ps: List[float], position: str) -> dict[str, float]:
    """Compute percentiles with calibrated sigma for better coverage.
    
    Inflates sigma based on position to match empirical prediction intervals.
    """
    config = RB_CONFIG if position == "RB" else WR_CONFIG
    calibration = config["sigma_calibration"]
    sigma_cal = sigma * calibration
    scale = float(np.exp(mu))
    return {f"p{int(p*100)}": float(lognorm.ppf(p, sigma_cal, scale=scale)) for p in ps}


def _pfr_fetch_rb_roster(season: int) -> List[Tuple[str, str]]:
    """Return list of (pfr_player_id, player_name) for RBs across all teams in season."""
    rbs: List[Tuple[str, str]] = []
    for tm in PFR_TEAMS:
        url = f"{PFR_BASE}/teams/{tm}/{season}_roster.htm"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = None
        # Find a table containing player and pos columns
        for t in soup.find_all("table"):
            header_cells = [th.get("data-stat") for th in t.find("thead").find_all("th")] if t.find("thead") else []
            if "player" in header_cells and "pos" in header_cells:
                table = t
                break
        if not table:
            continue
        tbody = table.find("tbody")
        for tr in tbody.find_all("tr"):
            pos_td = tr.find("td", {"data-stat": "pos"})
            if not pos_td:
                continue
            pos = (pos_td.get_text() or "").strip()
            if pos != "RB":
                continue
            player_th = tr.find("th", {"data-stat": "player"})
            if not player_th:
                continue
            pfr_id = player_th.get("data-append-csv")
            a = player_th.find("a")
            name = a.get_text().strip() if a else (player_th.get_text() or "").strip()
            if pfr_id:
                rbs.append((pfr_id, name))
    # Deduplicate by id
    seen = set()
    unique: List[Tuple[str, str]] = []
    for pid, nm in rbs:
        if pid not in seen:
            unique.append((pid, nm))
            seen.add(pid)
    return unique


def _pfr_fetch_player_gamelog_rushing(pfr_id: str, season: int) -> pd.DataFrame:
    """Fetch a single player's game log rushing yards for season (regular season only)."""
    first = pfr_id[0]
    url = f"{PFR_BASE}/players/{first}/{pfr_id}/gamelog/{season}/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    # Find table having rush_yds column
    table = None
    for t in soup.find_all("table"):
        thead = t.find("thead")
        if not thead:
            continue
        headers = [th.get("data-stat") for th in thead.find_all("th")]
        if "rush_yds" in headers and "week_num" in headers:
            table = t
            break
    if table is None:
        return pd.DataFrame(columns=["week", "season", "rushing_yards"])  # empty
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        # Skip separator/header rows
        if tr.get("class") and ("thead" in tr.get("class") or "divider" in tr.get("class")):
            continue
        week_td = tr.find("td", {"data-stat": "week_num"})
        if not week_td:
            continue
        week_txt = (week_td.get_text() or "").strip()
        if not week_txt.isdigit():
            # skip playoffs or non-regular rows
            continue
        week = int(week_txt)
        rush_td = tr.find("td", {"data-stat": "rush_yds"})
        ytxt = (rush_td.get_text() if rush_td else "").strip()
        try:
            yards = float(ytxt) if ytxt not in {"", "-"} else 0.0
        except ValueError:
            yards = 0.0
        rows.append({"week": week, "season": season, "rushing_yards": max(yards, 0.0)})
    return pd.DataFrame(rows)


def _pfr_fetch_weekly_rb_rushing(season: int) -> pd.DataFrame:
    """Fetch weekly rushing for all RBs (per game) via PFR gamelogs."""
    rb_list = _pfr_fetch_rb_roster(season)
    all_rows: List[dict] = []
    for pid, name in rb_list:
        df = _pfr_fetch_player_gamelog_rushing(pid, season)
        if df.empty:
            continue
        df = df.assign(player_id=pid, player_name=name, position="RB", season_type="REG")
        all_rows.append(df)
    if not all_rows:
        return pd.DataFrame(columns=["player_id", "player_name", "position", "season", "season_type", "week", "rushing_yards"])
    out = pd.concat(all_rows, ignore_index=True)
    # Rename to match nfl_data_py columns subset we use
    out["player_display_name"] = out["player_name"]
    return out[["player_id", "player_name", "player_display_name", "position", "season", "season_type", "week", "rushing_yards"]]


def _pfr_fetch_team_defense_allowed(season: int) -> pd.DataFrame:
    """Fetch per-team, per-week rushing yards allowed via team game log pages."""
    rows: List[dict] = []
    for pfr_tm, std_tm in PFR_TO_STD.items():
        url = f"{PFR_BASE}/teams/{pfr_tm}/{season}.htm"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = None
        for t in soup.find_all("table"):
            thead = t.find("thead")
            if not thead:
                continue
            headers = [th.get("data-stat") for th in thead.find_all("th")]
            if "opp_rush_yds" in headers and "week_num" in headers:
                table = t
                break
        if not table:
            continue
        for tr in table.find("tbody").find_all("tr"):
            if tr.get("class") and ("thead" in tr.get("class") or "divider" in tr.get("class")):
                continue
            week_td = tr.find("th", {"data-stat": "week_num"}) or tr.find("td", {"data-stat": "week_num"})
            if not week_td:
                continue
            wtxt = (week_td.get_text() or "").strip()
            if not wtxt.isdigit():
                continue
            week = int(wtxt)
            ytd = tr.find("td", {"data-stat": "opp_rush_yds"})
            ytxt = (ytd.get_text() if ytd else "").strip()
            try:
                yards = float(ytxt) if ytxt not in {"", "-"} else 0.0
            except ValueError:
                yards = 0.0
            rows.append({"team": std_tm, "week": week, "rush_allowed": max(yards, 0.0), "season": season, "season_type": "REG"})
    return pd.DataFrame(rows)


@app.command()
def fetch(
    season: int = typer.Option(2025, help="Season year"),
    season_type: str = typer.Option("REG", help="Season type: REG or POST"),
) -> None:
    """Fetch weekly player and team data for the season and persist to data/raw."""
    _ensure_dirs()
    console.print("[bold]Fetching 2025 data via nflreadpy...[/bold]")
    try:
        pbp = nr.load_pbp(seasons=[season])
        # Convert to pandas for aggregation simplicity
        pbp_df = pbp.to_pandas()
        pbp_df = pbp_df[(pbp_df["season"] == season) & (pbp_df["season_type"] == season_type)]

        # Weekly RB rushing yards per player
        rush_df = pbp_df[pbp_df["rush"] == 1]
        weekly_rb = rush_df.groupby([
            "season", "week", "rusher_player_id", "rusher_player_name", "posteam", "defteam"
        ], as_index=False)["yards_gained"].sum().rename(columns={"yards_gained": "rushing_yards"})

        # Join positions from rosters to filter RBs
        rosters = nr.load_rosters(seasons=[season]).to_pandas()
        rosters = rosters[["gsis_id", "position"]].drop_duplicates()
        weekly_rb = weekly_rb.merge(rosters, left_on="rusher_player_id", right_on="gsis_id", how="left")
        weekly_rb = weekly_rb[weekly_rb["position"] == "RB"]

        weekly_out = weekly_rb.assign(
            player_id=lambda d: d["rusher_player_id"],
            player_name=lambda d: d["rusher_player_name"],
            player_display_name=lambda d: d["rusher_player_name"],
            recent_team=lambda d: d["posteam"],
            opponent_team=lambda d: d["defteam"],
            season_type=season_type,
            position="RB",
        )[
            [
                "player_id", "player_name", "player_display_name", "position", "recent_team",
                "season", "season_type", "week", "opponent_team", "rushing_yards"
            ]
        ]
        weekly_out.to_parquet(RAW_DIR / f"weekly_RB_{season}_{season_type}.parquet")
        console.print("[green]Saved weekly RB rushing via nflreadpy[/green]")

        # Weekly WR receiving yards per player
        pass_df = pbp_df[pbp_df["pass"] == 1]
        # Filter to complete passes only
        complete_df = pass_df[pass_df["complete_pass"] == 1]
        weekly_wr = complete_df.groupby([
            "season", "week", "receiver_player_id", "receiver_player_name", "posteam", "defteam"
        ], as_index=False)["yards_gained"].sum().rename(columns={"yards_gained": "receiving_yards"})
        
        # Join positions from rosters to filter WRs
        weekly_wr = weekly_wr.merge(rosters, left_on="receiver_player_id", right_on="gsis_id", how="left")
        weekly_wr = weekly_wr[weekly_wr["position"] == "WR"]
        
        weekly_wr_out = weekly_wr.assign(
            player_id=lambda d: d["receiver_player_id"],
            player_name=lambda d: d["receiver_player_name"],
            player_display_name=lambda d: d["receiver_player_name"],
            recent_team=lambda d: d["posteam"],
            opponent_team=lambda d: d["defteam"],
            season_type=season_type,
            position="WR",
        )[
            [
                "player_id", "player_name", "player_display_name", "position", "recent_team",
                "season", "season_type", "week", "opponent_team", "receiving_yards"
            ]
        ]
        weekly_wr_out.to_parquet(RAW_DIR / f"weekly_WR_{season}_{season_type}.parquet")
        console.print("[green]Saved weekly WR receiving via nflreadpy[/green]")

        # Team rushing yards allowed per game (sum opponent rushing)
        allowed = rush_df.groupby(["season", "week", "defteam"], as_index=False)["yards_gained"].sum()
        allowed = allowed.rename(columns={"defteam": "team", "yards_gained": "rush_allowed"})
        allowed["season_type"] = season_type
        allowed.to_parquet(RAW_DIR / f"team_rush_allowed_{season}_{season_type}.parquet")
        console.print("[green]Saved team rush defense allowed via nflreadpy[/green]")
        
        # Team passing yards allowed per game (sum opponent passing)
        pass_allowed = complete_df.groupby(["season", "week", "defteam"], as_index=False)["yards_gained"].sum()
        pass_allowed = pass_allowed.rename(columns={"defteam": "team", "yards_gained": "pass_allowed"})
        pass_allowed["season_type"] = season_type
        pass_allowed.to_parquet(RAW_DIR / f"team_pass_allowed_{season}_{season_type}.parquet")
        console.print("[green]Saved team pass defense allowed via nflreadpy[/green]")
        
        # === NEW: Calculate usage metrics for RBs ===
        console.print("[bold]Calculating RB usage metrics...[/bold]")
        
        # Count snaps per player per game
        rb_snaps = pbp_df[pbp_df["rusher_player_id"].notna()].groupby([
            "season", "week", "rusher_player_id", "posteam"
        ]).size().reset_index(name="snap_count")
        
        # Count total team snaps per game
        team_snaps = pbp_df.groupby(["season", "week", "posteam"]).size().reset_index(name="team_snaps")
        
        # Merge to get snap percentage
        rb_snaps = rb_snaps.merge(team_snaps, on=["season", "week", "posteam"], how="left")
        rb_snaps["snap_pct"] = rb_snaps["snap_count"] / rb_snaps["team_snaps"]
        
        # Count rush attempts and targets for RBs
        rb_rushes = rush_df[rush_df["rusher_player_id"].notna()].groupby([
            "season", "week", "rusher_player_id", "posteam"
        ]).size().reset_index(name="rush_attempts")
        
        rb_targets = pass_df[pass_df["receiver_player_id"].notna()].groupby([
            "season", "week", "receiver_player_id", "posteam"
        ]).size().reset_index(name="targets")
        
        # Merge with roster to filter to RBs only
        rb_snaps_filt = rb_snaps.merge(rosters, left_on="rusher_player_id", right_on="gsis_id", how="left")
        rb_snaps_filt = rb_snaps_filt[rb_snaps_filt["position"] == "RB"]
        
        # Merge usage metrics
        rb_usage = rb_snaps_filt[["season", "week", "rusher_player_id", "posteam", "snap_count", "snap_pct"]].copy()
        rb_usage = rb_usage.merge(rb_rushes, left_on=["season", "week", "rusher_player_id", "posteam"],
                                   right_on=["season", "week", "rusher_player_id", "posteam"], how="left")
        rb_usage = rb_usage.merge(rb_targets, left_on=["season", "week", "rusher_player_id"],
                                   right_on=["season", "week", "receiver_player_id"], how="left",
                                   suffixes=("", "_target"))
        
        # Drop duplicate posteam columns from merge
        if "posteam_target" in rb_usage.columns:
            rb_usage = rb_usage.drop(columns=["posteam_target"])
        
        rb_usage["rush_attempts"] = rb_usage["rush_attempts"].fillna(0)
        rb_usage["targets"] = rb_usage["targets"].fillna(0)
        rb_usage["touches"] = rb_usage["rush_attempts"] + rb_usage["targets"]
        
        # Calculate team-level touch share
        team_touches = rb_usage.groupby(["season", "week", "posteam"])["touches"].sum().reset_index(name="team_touches")
        rb_usage = rb_usage.merge(team_touches, on=["season", "week", "posteam"], how="left")
        rb_usage["touch_share"] = rb_usage["touches"] / rb_usage["team_touches"].clip(lower=1)
        
        rb_usage_out = rb_usage.assign(
            player_id=lambda d: d["rusher_player_id"],
            recent_team=lambda d: d["posteam"],
            season_type=season_type,
        )[["player_id", "season", "season_type", "week", "recent_team", 
           "snap_count", "snap_pct", "rush_attempts", "targets", "touches", "touch_share"]]
        
        rb_usage_out.to_parquet(RAW_DIR / f"weekly_RB_usage_{season}_{season_type}.parquet")
        console.print("[green]Saved RB usage metrics[/green]")
        
        # === NEW: Calculate usage metrics for WRs ===
        console.print("[bold]Calculating WR usage metrics...[/bold]")
        
        # Count snaps for WRs (approximate by plays where they're involved)
        wr_snaps = pbp_df[pbp_df["receiver_player_id"].notna()].groupby([
            "season", "week", "receiver_player_id", "posteam"
        ]).size().reset_index(name="snap_count")
        
        wr_snaps = wr_snaps.merge(team_snaps, on=["season", "week", "posteam"], how="left")
        wr_snaps["snap_pct"] = wr_snaps["snap_count"] / wr_snaps["team_snaps"]
        
        # Filter to WRs only
        wr_snaps_filt = wr_snaps.merge(rosters, left_on="receiver_player_id", right_on="gsis_id", how="left")
        wr_snaps_filt = wr_snaps_filt[wr_snaps_filt["position"] == "WR"]
        
        # WR targets
        wr_targets_agg = pass_df[pass_df["receiver_player_id"].notna()].groupby([
            "season", "week", "receiver_player_id", "posteam"
        ]).size().reset_index(name="targets")
        
        # Calculate team target share
        team_targets = wr_targets_agg.groupby(["season", "week", "posteam"])["targets"].sum().reset_index(name="team_targets")
        wr_targets_agg = wr_targets_agg.merge(team_targets, on=["season", "week", "posteam"], how="left")
        wr_targets_agg["target_share"] = wr_targets_agg["targets"] / wr_targets_agg["team_targets"].clip(lower=1)
        
        # Air yards share (if available)
        if "air_yards" in pass_df.columns:
            wr_air = pass_df[pass_df["receiver_player_id"].notna()].groupby([
                "season", "week", "receiver_player_id", "posteam"
            ])["air_yards"].sum().reset_index(name="air_yards")
            
            team_air = wr_air.groupby(["season", "week", "posteam"])["air_yards"].sum().reset_index(name="team_air_yards")
            wr_air = wr_air.merge(team_air, on=["season", "week", "posteam"], how="left")
            wr_air["air_yards_share"] = wr_air["air_yards"] / wr_air["team_air_yards"].clip(lower=1)
        else:
            wr_air = pd.DataFrame()
        
        # Merge WR usage metrics
        wr_usage = wr_snaps_filt[["season", "week", "receiver_player_id", "posteam", "snap_count", "snap_pct"]].copy()
        wr_usage = wr_usage.merge(wr_targets_agg[["season", "week", "receiver_player_id", "posteam", "targets", "target_share"]], 
                                  on=["season", "week", "receiver_player_id", "posteam"], how="left")
        
        if not wr_air.empty:
            wr_usage = wr_usage.merge(wr_air[["season", "week", "receiver_player_id", "air_yards", "air_yards_share"]], 
                                      on=["season", "week", "receiver_player_id"], how="left")
        else:
            wr_usage["air_yards"] = 0
            wr_usage["air_yards_share"] = 0
        
        wr_usage["targets"] = wr_usage["targets"].fillna(0)
        wr_usage["target_share"] = wr_usage["target_share"].fillna(0)
        
        wr_usage_out = wr_usage.assign(
            player_id=lambda d: d["receiver_player_id"],
            recent_team=lambda d: d["posteam"],
            season_type=season_type,
        )[["player_id", "season", "season_type", "week", "recent_team", 
           "snap_count", "snap_pct", "targets", "target_share", "air_yards", "air_yards_share"]]
        
        wr_usage_out.to_parquet(RAW_DIR / f"weekly_WR_usage_{season}_{season_type}.parquet")
        console.print("[green]Saved WR usage metrics[/green]")
        
    except Exception as e:
        console.print(f"[red]nflreadpy failed for {season}: {e}")
        raise


@app.command()
def fit_players(
    season: int = typer.Option(2025),
    season_type: str = typer.Option("REG"),
    position_filter: str = typer.Option("RB", help="Filter to position group (e.g., RB, WR)"),
    use_weighting: bool = typer.Option(True, help="Use recency weighting (exponential decay)"),
    use_usage_filter: bool = typer.Option(True, help="Filter players by minimum usage thresholds"),
) -> None:
    """Fit per-player lognormal params (mu, sigma) from weekly yards with improvements."""
    _ensure_dirs()
    
    # Get position-specific config
    config = RB_CONFIG if position_filter == "RB" else WR_CONFIG if position_filter == "WR" else None
    if config is None:
        console.print(f"[red]Unsupported position: {position_filter}. Use RB or WR.[/red]")
        raise SystemExit(1)
    
    console.print(f"[bold]Fitting {position_filter} players with improved algorithm[/bold]")
    console.print(f"  Recency weighting: {use_weighting} (decay={config['recency_decay']})")
    console.print(f"  Usage filtering: {use_usage_filter}")
    console.print(f"  Sigma adjustment: {config['sigma_adjust']}x")
    
    # Determine yards column and allowed file based on position
    if position_filter == "RB":
        yards_col = "rushing_yards"
        allowed_col = "rush_allowed"
        allowed_path = RAW_DIR / f"team_rush_allowed_{season}_{season_type}.parquet"
        usage_path = RAW_DIR / f"weekly_RB_usage_{season}_{season_type}.parquet"
    elif position_filter == "WR":
        yards_col = "receiving_yards"
        allowed_col = "pass_allowed"
        allowed_path = RAW_DIR / f"team_pass_allowed_{season}_{season_type}.parquet"
        usage_path = RAW_DIR / f"weekly_WR_usage_{season}_{season_type}.parquet"
    else:
        console.print(f"[red]Unsupported position: {position_filter}.[/red]")
        raise SystemExit(1)
    
    weekly_path = RAW_DIR / f"weekly_{position_filter}_{season}_{season_type}.parquet"
    if not weekly_path.exists():
        console.print(f"[red]Missing weekly data at {weekly_path}. Run fetch first.[/red]")
        raise SystemExit(1)

    weekly = pd.read_parquet(weekly_path)
    weekly = weekly[(weekly["season"] == season) & (weekly["season_type"] == season_type)]
    weekly = weekly[weekly["position"] == position_filter]
    
    # Load usage data if available and filtering is enabled
    usage_df = None
    if use_usage_filter and usage_path.exists():
        usage_df = pd.read_parquet(usage_path)
        usage_df = usage_df[(usage_df["season"] == season) & (usage_df["season_type"] == season_type)]
        console.print(f"[green]Loaded usage data: {len(usage_df)} player-week records[/green]")

    # Opponent normalization: scale each game's yards by opponent's season-to-date defensive strength
    if allowed_path.exists() and not weekly.empty and "opponent_team" in weekly.columns:
        allowed = pd.read_parquet(allowed_path)
        allowed = allowed[(allowed["season"] == season) & (allowed["season_type"] == season_type)]
        allowed = allowed[["team", "week", allowed_col]].copy()
        # Team cumulative mean up to prior week
        allowed = allowed.sort_values(["team", "week"])
        allowed["team_cum_sum"] = allowed.groupby("team")[allowed_col].cumsum().shift(1)
        allowed["team_cum_cnt"] = allowed.groupby("team").cumcount()
        allowed["team_cum_mean_prev"] = allowed["team_cum_sum"] / allowed["team_cum_cnt"].replace(0, np.nan)
        # League cumulative mean up to prior week
        wk_mean = (
            allowed.groupby("week")[allowed_col].mean().reset_index().sort_values("week")
        )
        wk_mean["cum_sum"] = wk_mean[allowed_col].cumsum().shift(1)
        wk_mean["cum_cnt"] = np.arange(1, len(wk_mean) + 1) - 1
        wk_mean["league_cum_mean_prev"] = wk_mean["cum_sum"] / wk_mean["cum_cnt"].replace(0, np.nan)
        allowed = allowed.merge(wk_mean[["week", "league_cum_mean_prev"]], on="week", how="left")
        eps = 1e-8
        allowed["norm_mult"] = (
            allowed["league_cum_mean_prev"].fillna(allowed[allowed_col].mean())
            / allowed["team_cum_mean_prev"].fillna(allowed[allowed_col].mean()).clip(lower=eps)
        )
        mult = allowed[["team", "week", "norm_mult"]]
        weekly = weekly.merge(mult, left_on=["opponent_team", "week"], right_on=["team", "week"], how="left")
        weekly["yards_adj"] = (weekly[yards_col] * weekly["norm_mult"].fillna(1.0)).clip(lower=0)
    else:
        weekly["yards_adj"] = weekly[yards_col].clip(lower=0)

    # Merge with usage data
    if usage_df is not None:
        weekly = weekly.merge(usage_df, on=["player_id", "season", "season_type", "week"], how="left")

    # Densify weeks: ensure every player has an entry for each week
    if not weekly.empty:
        all_weeks = sorted(weekly["week"].unique().tolist())
        player_meta = weekly.groupby("player_id").agg({
            "player_name": "first",
            "player_display_name": "first",
        }).reset_index()

        # Build per-player week grid
        grids = []
        for pid, grp in weekly.groupby("player_id"):
            present = grp[["week", "yards_adj"]].set_index("week")
            reindexed = present.reindex(all_weeks, fill_value=0).reset_index().rename(columns={"index": "week"})
            reindexed.insert(0, "player_id", pid)
            
            # If usage data available, merge it back
            if usage_df is not None and "snap_pct" in grp.columns:
                usage_cols = ["week"] + [c for c in grp.columns if c in ["snap_pct", "touch_share", "target_share", "touches", "targets"]]
                usage_present = grp[usage_cols].set_index("week")
                usage_reindexed = usage_present.reindex(all_weeks, fill_value=0).reset_index()
                reindexed = reindexed.merge(usage_reindexed, on="week", how="left")
            
            grids.append(reindexed)
        dense = pd.concat(grids, ignore_index=True)
        weekly = dense.merge(player_meta, on="player_id", how="left")
        weekly["season"] = season
        weekly["season_type"] = season_type
        weekly["position"] = position_filter

    # Fit models for each player
    results = []
    filtered_count = 0
    
    for player_id, grp in weekly.groupby("player_id"):
        player_name = grp["player_display_name"].iloc[0] if "player_display_name" in grp else grp["player_name"].iloc[0]
        yards = grp["yards_adj"].fillna(0).clip(lower=0)
        weeks = grp["week"]
        
        # Apply usage filtering
        if use_usage_filter and usage_df is not None and "snap_pct" in grp.columns:
            games_played = (yards > 0).sum()
            
            if position_filter == "RB":
                # Check snap percentage and touches
                recent_snaps = grp[grp["snap_pct"] > 0]["snap_pct"].tail(4)  # Last 4 games
                avg_snap_pct = recent_snaps.mean() if len(recent_snaps) > 0 else 0
                
                avg_touches = grp[grp["touches"] > 0]["touches"].mean() if "touches" in grp.columns and (grp["touches"] > 0).any() else 0
                
                # Filter criteria: min games, min snap %, min touches
                # Relax filtering: Only filter if ALL criteria fail badly
                if games_played < config["min_games"] and avg_snap_pct < (config["min_snap_pct"] / 2) and avg_touches < (config["min_touches_per_game"] / 2):
                    filtered_count += 1
                    continue
            
            elif position_filter == "WR":
                # Check snap percentage or target share
                recent_snaps = grp[grp["snap_pct"] > 0]["snap_pct"].tail(4) if (grp["snap_pct"] > 0).any() else pd.Series([0])
                avg_snap_pct = recent_snaps.mean() if len(recent_snaps) > 0 else 0
                
                recent_targets = grp[grp["target_share"] > 0]["target_share"].tail(4) if "target_share" in grp.columns and (grp["target_share"] > 0).any() else pd.Series([0])
                avg_target_share = recent_targets.mean() if len(recent_targets) > 0 else 0
                
                # Filter criteria: min games AND (min snap % OR min target share)
                # Relax filtering: Only filter if really low usage
                if games_played < config["min_games"] and avg_snap_pct < (config["min_snap_pct"] / 2) and avg_target_share < (config["min_target_share"] / 2):
                    filtered_count += 1
                    continue
        
        # Fit lognormal with or without weighting
        if use_weighting:
            mu, sigma = _fit_lognormal_weighted(yards, weeks, decay_factor=config["recency_decay"])
        else:
            mu, sigma = _fit_lognormal_from_yards(yards)
        
        # Apply position-specific sigma adjustment
        sigma = sigma * config["sigma_adjust"]
        
        # Calculate metrics
        expected = _expected_from_params(mu, sigma)
        percentiles = _percentiles_from_params_calibrated(mu, sigma, [0.1, 0.25, 0.5, 0.75, 0.9], position_filter)
        
        # Store additional metadata
        result = {
            "player_id": player_id,
            "player_name": player_name,
            "mu": mu,
            "sigma": sigma,
            "expected": expected,
            "games_played": int((yards > 0).sum()),
            **percentiles,
        }
        
        # Add usage metrics if available
        if usage_df is not None:
            if position_filter == "RB" and "snap_pct" in grp.columns:
                result["avg_snap_pct"] = float(grp[grp["snap_pct"] > 0]["snap_pct"].mean()) if (grp["snap_pct"] > 0).any() else 0
                result["avg_touch_share"] = float(grp[grp["touch_share"] > 0]["touch_share"].mean()) if "touch_share" in grp.columns and (grp["touch_share"] > 0).any() else 0
            elif position_filter == "WR" and "target_share" in grp.columns:
                result["avg_snap_pct"] = float(grp[grp["snap_pct"] > 0]["snap_pct"].mean()) if "snap_pct" in grp.columns and (grp["snap_pct"] > 0).any() else 0
                result["avg_target_share"] = float(grp[grp["target_share"] > 0]["target_share"].mean()) if (grp["target_share"] > 0).any() else 0
        
        results.append(result)

    console.print(f"[yellow]Filtered out {filtered_count} players due to low usage[/yellow]")
    console.print(f"[green]Fit models for {len(results)} players[/green]")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"player_lognorm_{position_filter}_{season}_{season_type}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    console.print(f"Saved {position_filter} params -> {out_path}")


@app.command()
def defense_adjustments(
    season: int = typer.Option(2025),
    season_type: str = typer.Option("REG"),
    position: str = typer.Option("RB", help="Position to calculate defense for (RB or WR)"),
) -> None:
    """Compute team defensive yards allowed deltas vs league average in log space (season-to-date)."""
    _ensure_dirs()
    
    # Determine allowed column based on position
    if position == "RB":
        allowed_col = "rush_allowed"
        allowed_path = RAW_DIR / f"team_rush_allowed_{season}_{season_type}.parquet"
    elif position == "WR":
        allowed_col = "pass_allowed"
        allowed_path = RAW_DIR / f"team_pass_allowed_{season}_{season_type}.parquet"
    else:
        console.print(f"[red]Unsupported position: {position}. Use RB or WR.[/red]")
        raise SystemExit(1)
    
    if not allowed_path.exists():
        console.print(f"[red]Missing defense data at {allowed_path}. Run fetch first.[/red]")
        raise SystemExit(1)

    allowed = pd.read_parquet(allowed_path)
    allowed = allowed[(allowed["season"] == season) & (allowed["season_type"] == season_type)]
    df_allowed = allowed[["team", "week", allowed_col]].copy()

    # Compute season-to-date arithmetic averages, then take logs for delta
    team_avg = df_allowed.groupby("team")[allowed_col].mean().rename("team_avg").reset_index()
    league_avg = float(df_allowed[allowed_col].mean())
    # Multiplicative adjustment in log space: ln(team_avg / league_avg)
    eps = 1e-8
    team_avg["delta_log"] = np.log(team_avg["team_avg"].clip(lower=eps)) - np.log(max(league_avg, eps))
    team_delta = team_avg[["team", "delta_log"]]

    out = team_delta.to_dict(orient="records")
    out_path = PROCESSED_DIR / f"team_defense_delta_{position}_{season}_{season_type}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    console.print(f"Saved {position} defense deltas -> {out_path}")


@app.command()
def predict(
    season: int = typer.Option(2025),
    season_type: str = typer.Option("REG"),
    position: str = typer.Option("RB", help="Position (RB or WR)"),
    player_id: Optional[str] = typer.Option(None, help="nfl_data_py player_id"),
    player_name: Optional[str] = typer.Option(None, help="Player display name if id unknown"),
    opponent_team: str = typer.Option(..., help="Opponent team code, e.g., DAL, SF"),
    line: float = typer.Option(70.5, help="Betting line to compute over probability"),
) -> None:
    """Compute adjusted player distribution vs opponent and probability over a betting line."""
    _ensure_dirs()
    params_path = PROCESSED_DIR / f"player_lognorm_{position}_{season}_{season_type}.json"
    deltas_path = PROCESSED_DIR / f"team_defense_delta_{position}_{season}_{season_type}.json"
    if not params_path.exists() or not deltas_path.exists():
        console.print(f"[red]Missing processed params or deltas for {position}. Run fit_players and defense_adjustments.[/red]")
        raise SystemExit(1)

    params = pd.DataFrame(json.loads(params_path.read_text()))
    deltas = pd.DataFrame(json.loads(deltas_path.read_text()))

    if player_id:
        row = params[params["player_id"] == player_id]
    elif player_name:
        row = params[params["player_name"].str.lower() == player_name.lower()]
    else:
        console.print("[red]Provide player_id or player_name[/red]")
        raise SystemExit(1)

    if row.empty:
        console.print("[red]Player not found in params[/red]")
        raise SystemExit(1)

    mu = float(row.iloc[0]["mu"])
    sigma = float(row.iloc[0]["sigma"])

    # Find delta for opponent
    delta_row = deltas[deltas["team"] == opponent_team]
    if delta_row.empty:
        console.print(f"[yellow]No delta found for {opponent_team}, assuming 0[/yellow]")
        delta = 0.0
    else:
        delta = float(delta_row.iloc[0]["delta_log"])

    adj_mu = mu + delta
    scale = float(np.exp(adj_mu))
    prob_over = float(1 - lognorm.cdf(line, sigma, scale=scale))
    expected = _expected_from_params(adj_mu, sigma)
    percentiles = _percentiles_from_params(adj_mu, sigma, [0.1, 0.25, 0.5, 0.75, 0.9])

    result = {
        "player_id": row.iloc[0]["player_id"],
        "player_name": row.iloc[0]["player_name"],
        "opponent_team": opponent_team,
        "mu": adj_mu,
        "sigma": sigma,
        "expected": expected,
        **percentiles,
        "prob_over_line": prob_over,
        "line": line,
    }

    console.print_json(data=result)


@app.command()
def plot(
    season: int = typer.Option(2025),
    season_type: str = typer.Option("REG"),
    position: str = typer.Option("RB", help="Position (RB or WR)"),
    player_id: Optional[str] = typer.Option(None),
    player_name: Optional[str] = typer.Option(None),
    opponent_team: Optional[str] = typer.Option(None),
    line: Optional[float] = typer.Option(None),
) -> None:
    """Save a PNG plot of the (optionally adjusted) player distribution, with line if provided."""
    import matplotlib.pyplot as plt

    _ensure_dirs()
    params_path = PROCESSED_DIR / f"player_lognorm_{position}_{season}_{season_type}.json"
    deltas_path = PROCESSED_DIR / f"team_defense_delta_{position}_{season}_{season_type}.json"
    if not params_path.exists():
        fb = PROCESSED_DIR / f"player_lognorm_{position}_{season-1}_{season_type}.json"
        if fb.exists():
            params_path = fb
        else:
            console.print(f"[red]Missing {position} params JSON. Run fit_players first.[/red]")
            raise SystemExit(1)
    if opponent_team and not deltas_path.exists():
        fb2 = PROCESSED_DIR / f"team_defense_delta_{position}_{season-1}_{season_type}.json"
        if fb2.exists():
            deltas_path = fb2

    params = pd.DataFrame(json.loads(params_path.read_text()))
    if player_id:
        row = params[params["player_id"] == player_id]
    elif player_name:
        row = params[params["player_name"].str.lower() == player_name.lower()]
    else:
        console.print("[red]Provide player_id or player_name[/red]")
        raise SystemExit(1)
    if row.empty:
        console.print("[red]Player not found[/red]")
        raise SystemExit(1)

    mu = float(row.iloc[0]["mu"])
    sigma = float(row.iloc[0]["sigma"])
    name = str(row.iloc[0]["player_name"]).replace(" ", "_")
    adj_mu = mu
    if opponent_team and deltas_path.exists():
        deltas = pd.DataFrame(json.loads(deltas_path.read_text()))
        drow = deltas[deltas["team"] == opponent_team]
        if not drow.empty:
            adj_mu = mu + float(drow.iloc[0]["delta_log"])

    scale = float(np.exp(adj_mu))
    xs = np.linspace(0, lognorm.ppf(0.98, sigma, scale=scale), 400)
    ys = lognorm.pdf(xs, sigma, scale=scale)

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label=f"{row.iloc[0]['player_name']} adj_mu={adj_mu:.2f} sigma={sigma:.2f}")
    if line is not None:
        plt.axvline(line, color="red", linestyle="--", label=f"line {line}")
        prob_over = 1 - lognorm.cdf(line, sigma, scale=scale)
        plt.text(line, max(ys)*0.8, f"P(>line)={prob_over:.2%}", rotation=90, color="red")
    plt.xlabel("Rushing yards")
    plt.ylabel("Density")
    plt.title("Lognormal fit")
    plt.legend()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{name}_{opponent_team or 'no_opp'}.png"
    plt.tight_layout()
    plt.savefig(out)
    console.print(f"Saved plot -> {out}")


@app.command()
def schedule_lines(
    season: int = typer.Option(2025),
    season_type: str = typer.Option("REG"),
    output: Optional[Path] = typer.Option(None, help="Output CSV path; defaults to reports/schedule_rb_lines_{season}_{season_type}.csv"),
) -> None:
    """Build a spreadsheet of all games with each RB's 50-50 predicted line (adjusted median)."""
    _ensure_dirs()
    params_path = PROCESSED_DIR / f"player_lognorm_{season}_{season_type}.json"
    deltas_path = PROCESSED_DIR / f"team_defense_delta_{season}_{season_type}.json"
    if not params_path.exists() or not deltas_path.exists():
        console.print("[red]Missing params or deltas. Run fit_players and defense_adjustments first.[/red]")
        raise SystemExit(1)

    params = pd.DataFrame(json.loads(params_path.read_text()))
    deltas = pd.DataFrame(json.loads(deltas_path.read_text()))

    # Load schedules
    sched = nr.load_schedules(seasons=[season])
    sched_pd = sched.to_pandas()
    sched_pd = sched_pd[sched_pd["game_type"] == season_type]

    # Load current rosters to map RBs to teams
    rost = nr.load_rosters(seasons=[season]).to_pandas()
    rb_rost = rost[rost["position"] == "RB"][["full_name", "gsis_id", "team"]].drop_duplicates()

    # We'll compute predicted median as adjusted mu: median = exp(mu) so adjust mu by opponent delta
    def get_delta(team: str) -> float:
        row = deltas[deltas["team"] == team]
        return float(row.iloc[0]["delta_log"]) if not row.empty else 0.0

    records: List[dict] = []
    for _, g in sched_pd.iterrows():
        home = g["home_team"]
        away = g["away_team"]
        week = int(g["week"])
        game_id = g["game_id"]

        # Home RBs face away defense; Away RBs face home defense
        home_delta = get_delta(away)
        away_delta = get_delta(home)

        for team, opp_delta in [(home, home_delta), (away, away_delta)]:
            team_rbs = rb_rost[rb_rost["team"] == team]
            for _, r in team_rbs.iterrows():
                name = r["full_name"]
                # Match by player_name in params (params uses short names like D.Henry). We'll try contains match as fallback.
                row = params[params["player_name"].str.contains(name.split(" ")[-1], case=False, na=False)]
                if row.empty:
                    continue
                mu = float(row.iloc[0]["mu"]) + opp_delta
                sigma = float(row.iloc[0]["sigma"])  # not used for median
                median = float(np.exp(mu))
                records.append({
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "team": team,
                    "opponent": home if team == away else away,
                    "player_name": name,
                    "predicted_line_50_50": round(median, 2),
                })

    out_df = pd.DataFrame(records)
    if output is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output = REPORTS_DIR / f"schedule_rb_lines_{season}_{season_type}.csv"
    out_df.to_csv(output, index=False)
    console.print(f"Saved schedule RB lines -> {output}")


@app.command()
def schedule_predictions(
    season: int = typer.Option(2025),
    season_type: str = typer.Option("REG"),
    week: Optional[int] = typer.Option(None, help="Specific week to generate predictions for; if None, all remaining weeks"),
    top_n: int = typer.Option(30, help="Number of top players to include per position"),
    output: Optional[Path] = typer.Option(None, help="Output CSV path; defaults to reports/predictions_{season}_{season_type}_week{week}.csv"),
) -> None:
    """Build comprehensive predictions spreadsheet with top N RBs and top N WRs for upcoming games."""
    _ensure_dirs()
    
    # Load params and deltas for both positions
    rb_params_path = PROCESSED_DIR / f"player_lognorm_RB_{season}_{season_type}.json"
    rb_deltas_path = PROCESSED_DIR / f"team_defense_delta_RB_{season}_{season_type}.json"
    wr_params_path = PROCESSED_DIR / f"player_lognorm_WR_{season}_{season_type}.json"
    wr_deltas_path = PROCESSED_DIR / f"team_defense_delta_WR_{season}_{season_type}.json"
    
    if not all([rb_params_path.exists(), rb_deltas_path.exists(), wr_params_path.exists(), wr_deltas_path.exists()]):
        console.print("[red]Missing params or deltas. Run fit_players and defense_adjustments for both RB and WR.[/red]")
        raise SystemExit(1)
    
    rb_params = pd.DataFrame(json.loads(rb_params_path.read_text()))
    rb_deltas = pd.DataFrame(json.loads(rb_deltas_path.read_text()))
    wr_params = pd.DataFrame(json.loads(wr_params_path.read_text()))
    wr_deltas = pd.DataFrame(json.loads(wr_deltas_path.read_text()))
    
    # Filter to top N for both positions by expected yards
    rb_params_sorted = rb_params.sort_values("expected", ascending=False).head(top_n)
    top_rb_ids = set(rb_params_sorted["player_id"].tolist())
    
    wr_params_sorted = wr_params.sort_values("expected", ascending=False).head(top_n)
    top_wr_ids = set(wr_params_sorted["player_id"].tolist())
    
    # Load schedules
    sched = nr.load_schedules(seasons=[season])
    sched_pd = sched.to_pandas()
    sched_pd = sched_pd[sched_pd["game_type"] == season_type]
    
    if week is not None:
        sched_pd = sched_pd[sched_pd["week"] == week]
    
    # Load current rosters
    rost = nr.load_rosters(seasons=[season]).to_pandas()
    rb_rost = rost[rost["position"] == "RB"][["full_name", "gsis_id", "team"]].drop_duplicates()
    wr_rost = rost[rost["position"] == "WR"][["full_name", "gsis_id", "team"]].drop_duplicates()
    
    def get_delta(team: str, deltas: pd.DataFrame) -> float:
        row = deltas[deltas["team"] == team]
        return float(row.iloc[0]["delta_log"]) if not row.empty else 0.0
    
    def find_player_params(player_gsis_id: str, player_name: str, params_df: pd.DataFrame) -> Optional[pd.Series]:
        # Try exact match first
        row = params_df[params_df["player_id"] == player_gsis_id]
        if not row.empty:
            return row.iloc[0]
        # Fallback: match by last name
        last_name = player_name.split(" ")[-1]
        row = params_df[params_df["player_name"].str.contains(last_name, case=False, na=False)]
        if not row.empty:
            return row.iloc[0]
        return None
    
    records: List[dict] = []
    
    for _, g in sched_pd.iterrows():
        home = g["home_team"]
        away = g["away_team"]
        wk = int(g["week"])
        game_id = g["game_id"]
        
        # Home players face away defense; Away players face home defense
        home_rb_delta = get_delta(away, rb_deltas)
        away_rb_delta = get_delta(home, rb_deltas)
        home_wr_delta = get_delta(away, wr_deltas)
        away_wr_delta = get_delta(home, wr_deltas)
        
        # Process top RBs
        for team, rb_delta in [(home, home_rb_delta), (away, away_rb_delta)]:
            team_rbs = rb_rost[rb_rost["team"] == team]
            for _, r in team_rbs.iterrows():
                # Only include top N RBs
                if r["gsis_id"] not in top_rb_ids:
                    continue
                    
                player_params = find_player_params(r["gsis_id"], r["full_name"], rb_params)
                if player_params is None:
                    continue
                
                mu = float(player_params["mu"]) + rb_delta
                sigma = float(player_params["sigma"])
                median = float(np.exp(mu))
                expected = _expected_from_params(mu, sigma)
                percentiles = _percentiles_from_params(mu, sigma, [0.25, 0.75])
                p25 = percentiles["p25"]
                p75 = percentiles["p75"]
                
                records.append({
                    "game_id": game_id,
                    "season": season,
                    "week": wk,
                    "position": "RB",
                    "player_id": r["gsis_id"],
                    "player_name": r["full_name"],
                    "team": team,
                    "opponent": home if team == away else away,
                    "predicted_p25": round(p25, 1),
                    "predicted_median": round(median, 1),
                    "predicted_p75": round(p75, 1),
                    "predicted_expected": round(expected, 1),
                })
        
        # Process top WRs
        for team, wr_delta in [(home, home_wr_delta), (away, away_wr_delta)]:
            team_wrs = wr_rost[wr_rost["team"] == team]
            for _, r in team_wrs.iterrows():
                # Only include top N WRs
                if r["gsis_id"] not in top_wr_ids:
                    continue
                    
                player_params = find_player_params(r["gsis_id"], r["full_name"], wr_params)
                if player_params is None:
                    continue
                
                mu = float(player_params["mu"]) + wr_delta
                sigma = float(player_params["sigma"])
                median = float(np.exp(mu))
                expected = _expected_from_params(mu, sigma)
                percentiles = _percentiles_from_params(mu, sigma, [0.25, 0.75])
                p25 = percentiles["p25"]
                p75 = percentiles["p75"]
                
                records.append({
                    "game_id": game_id,
                    "season": season,
                    "week": wk,
                    "position": "WR",
                    "player_id": r["gsis_id"],
                    "player_name": r["full_name"],
                    "team": team,
                    "opponent": home if team == away else away,
                    "predicted_p25": round(p25, 1),
                    "predicted_median": round(median, 1),
                    "predicted_p75": round(p75, 1),
                    "predicted_expected": round(expected, 1),
                })
    
    # Create dataframe and split by position
    all_df = pd.DataFrame(records)
    rb_df = all_df[all_df["position"] == "RB"].sort_values(["week", "predicted_expected"], ascending=[True, False])
    wr_df = all_df[all_df["position"] == "WR"].sort_values(["week", "predicted_expected"], ascending=[True, False])
    
    # Determine output paths
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    week_str = f"week{week}" if week else "all_weeks"
    
    if output is None:
        rb_output = REPORTS_DIR / f"predictions_RB_{season}_{season_type}_{week_str}.csv"
        wr_output = REPORTS_DIR / f"predictions_WR_{season}_{season_type}_{week_str}.csv"
    else:
        # If custom output provided, append position to filename
        base = output.stem
        ext = output.suffix
        parent = output.parent
        rb_output = parent / f"{base}_RB{ext}"
        wr_output = parent / f"{base}_WR{ext}"
    
    # Save both files
    rb_df.to_csv(rb_output, index=False)
    wr_df.to_csv(wr_output, index=False)
    
    console.print(f"[green]Saved RB predictions ({len(rb_df)} player-games) to {rb_output}[/green]")
    console.print(f"[green]Saved WR predictions ({len(wr_df)} player-games) to {wr_output}[/green]")
    console.print(f"  Top {top_n} players per position included")


@app.command()
def backtest(
    season: int = typer.Option(2025),
    season_type: str = typer.Option("REG"),
    train_weeks_str: str = typer.Option("1,2,3,4", help="Comma-separated training weeks"),
    test_weeks_str: str = typer.Option("5,6", help="Comma-separated test weeks"),
    position: str = typer.Option("RB", help="Position to backtest (RB or WR)"),
    use_weighting: bool = typer.Option(True, help="Use recency weighting"),
    use_usage_filter: bool = typer.Option(True, help="Apply usage filtering"),
) -> None:
    """Backtest predictions by training on train_weeks and evaluating on test_weeks."""
    _ensure_dirs()
    
    console.print(f"[bold]Backtesting {position} predictions[/bold]")
    
    # Parse week ranges
    train_weeks = [int(w.strip()) for w in train_weeks_str.split(",")]
    test_weeks = [int(w.strip()) for w in test_weeks_str.split(",")]
    
    console.print(f"Training weeks: {train_weeks}")
    console.print(f"Test weeks: {test_weeks}")
    
    # Determine file paths
    if position == "RB":
        yards_col = "rushing_yards"
        weekly_path = RAW_DIR / f"weekly_RB_{season}_{season_type}.parquet"
        usage_path = RAW_DIR / f"weekly_RB_usage_{season}_{season_type}.parquet"
    elif position == "WR":
        yards_col = "receiving_yards"
        weekly_path = RAW_DIR / f"weekly_WR_{season}_{season_type}.parquet"
        usage_path = RAW_DIR / f"weekly_WR_usage_{season}_{season_type}.parquet"
    else:
        console.print(f"[red]Unsupported position: {position}[/red]")
        raise SystemExit(1)
    
    if not weekly_path.exists():
        console.print(f"[red]Missing weekly data at {weekly_path}. Run fetch first.[/red]")
        raise SystemExit(1)
    
    # Load full data
    weekly_full = pd.read_parquet(weekly_path)
    weekly_full = weekly_full[(weekly_full["season"] == season) & (weekly_full["season_type"] == season_type)]
    
    # Load usage data if available
    usage_df = None
    if usage_path.exists():
        usage_df = pd.read_parquet(usage_path)
        usage_df = usage_df[(usage_df["season"] == season) & (usage_df["season_type"] == season_type)]
    
    # Split into train and test
    train_data = weekly_full[weekly_full["week"].isin(train_weeks)].copy()
    test_data = weekly_full[weekly_full["week"].isin(test_weeks)].copy()
    
    console.print(f"Train data: {len(train_data)} records")
    console.print(f"Test data: {len(test_data)} records")
    
    # Train models on train_weeks using simplified logic
    console.print("[bold]Training models...[/bold]")
    
    config = RB_CONFIG if position == "RB" else WR_CONFIG
    player_models = {}
    
    for player_id, grp in train_data.groupby("player_id"):
        yards = grp[yards_col].fillna(0).clip(lower=0)
        weeks = grp["week"]
        
        # Apply usage filtering if enabled (relaxed for backtesting)
        if use_usage_filter and usage_df is not None:
            usage_grp = usage_df[usage_df["player_id"] == player_id]
            if not usage_grp.empty:
                games_played = (yards > 0).sum()
                # Relax min_games check for short training periods
                min_games_threshold = min(config["min_games"], len(train_weeks) - 1)
                if games_played < min_games_threshold:
                    continue
                
                if position == "RB" and "snap_pct" in usage_grp.columns:
                    avg_snap_pct = usage_grp[usage_grp["snap_pct"] > 0]["snap_pct"].mean() if (usage_grp["snap_pct"] > 0).any() else 0
                    avg_touches = usage_grp["touches"].mean() if "touches" in usage_grp.columns else 0
                    # Relaxed thresholds for backtesting
                    if avg_snap_pct < (config["min_snap_pct"] / 2) and avg_touches < (config["min_touches_per_game"] / 2):
                        continue
                elif position == "WR" and "target_share" in usage_grp.columns:
                    avg_snap_pct = usage_grp[usage_grp["snap_pct"] > 0]["snap_pct"].mean() if "snap_pct" in usage_grp.columns and (usage_grp["snap_pct"] > 0).any() else 0
                    avg_target_share = usage_grp[usage_grp["target_share"] > 0]["target_share"].mean() if (usage_grp["target_share"] > 0).any() else 0
                    # Relaxed thresholds for backtesting
                    if avg_snap_pct < (config["min_snap_pct"] / 2) and avg_target_share < (config["min_target_share"] / 2):
                        continue
        
        # Fit model
        if use_weighting:
            mu, sigma = _fit_lognormal_weighted(yards, weeks, decay_factor=config["recency_decay"])
        else:
            mu, sigma = _fit_lognormal_from_yards(yards)
        
        sigma = sigma * config["sigma_adjust"]
        
        player_models[player_id] = {
            "mu": mu,
            "sigma": sigma,
            "player_name": grp["player_name"].iloc[0] if "player_name" in grp.columns else grp["player_display_name"].iloc[0]
        }
    
    console.print(f"[green]Trained {len(player_models)} player models[/green]")
    
    # Generate predictions for test weeks
    console.print("[bold]Generating predictions for test weeks...[/bold]")
    
    predictions = []
    actuals = []
    
    for _, row in test_data.iterrows():
        player_id = row["player_id"]
        if player_id not in player_models:
            continue
        
        model = player_models[player_id]
        mu = model["mu"]
        sigma = model["sigma"]
        
        # Simple prediction without defensive adjustment for now
        predicted_median = float(np.exp(mu))
        predicted_expected = _expected_from_params(mu, sigma)
        predicted_p75 = _percentiles_from_params_calibrated(mu, sigma, [0.75], position)["p75"]
        
        actual_yards = float(row[yards_col])
        
        predictions.append({
            "player_id": player_id,
            "player_name": model["player_name"],
            "week": int(row["week"]),
            "predicted_median": predicted_median,
            "predicted_expected": predicted_expected,
            "predicted_p75": predicted_p75,
            "actual_yards": actual_yards,
        })
        
        actuals.append(actual_yards)
    
    if not predictions:
        console.print("[red]No predictions generated. Check that players exist in both train and test sets.[/red]")
        raise SystemExit(1)
    
    # Calculate evaluation metrics
    pred_df = pd.DataFrame(predictions)
    actual_arr = np.array(actuals)
    predicted_expected_arr = pred_df["predicted_expected"].values
    predicted_median_arr = pred_df["predicted_median"].values
    predicted_p75_arr = pred_df["predicted_p75"].values
    
    mae = np.mean(np.abs(actual_arr - predicted_expected_arr))
    rmse = np.sqrt(np.mean((actual_arr - predicted_expected_arr) ** 2))
    mape = np.mean(np.abs((actual_arr - predicted_expected_arr) / (actual_arr + 1))) * 100
    correlation = np.corrcoef(actual_arr, predicted_expected_arr)[0, 1]
    
    # Directional accuracy
    league_median = np.median(actual_arr)
    pred_over_median = predicted_expected_arr > league_median
    actual_over_median = actual_arr > league_median
    directional_accuracy = np.mean(pred_over_median == actual_over_median) * 100
    
    # Coverage (% of actuals within median to p75)
    within_interval = np.sum((actual_arr >= predicted_median_arr) & (actual_arr <= predicted_p75_arr))
    coverage = (within_interval / len(actual_arr)) * 100
    
    # Print results
    console.print("\n" + "=" * 60)
    console.print(f"[bold green]BACKTEST RESULTS: {position} (Train: weeks {train_weeks}, Test: weeks {test_weeks})[/bold green]")
    console.print("=" * 60)
    console.print(f"\nPredictions generated: {len(predictions)}")
    console.print(f"\n[bold]Performance Metrics:[/bold]")
    console.print(f"  MAE:                    {mae:.2f} yards")
    console.print(f"  RMSE:                   {rmse:.2f} yards")
    console.print(f"  MAPE:                   {mape:.2f}%")
    console.print(f"  Correlation:            {correlation:.3f}")
    console.print(f"  Directional Accuracy:   {directional_accuracy:.1f}%")
    console.print(f"  Coverage (50-75th):     {coverage:.1f}%")
    console.print("=" * 60)
    
    # Save results
    output_file = REPORTS_DIR / f"backtest_{position}_{season}_{season_type}_train{min(train_weeks)}-{max(train_weeks)}_test{min(test_weeks)}-{max(test_weeks)}.csv"
    pred_df.to_csv(output_file, index=False)
    console.print(f"\n[green]Saved backtest results to {output_file}[/green]")
    
    # Save summary metrics
    metrics_dict = {
        "position": position,
        "season": season,
        "season_type": season_type,
        "train_weeks": str(train_weeks),
        "test_weeks": str(test_weeks),
        "use_weighting": use_weighting,
        "use_usage_filter": use_usage_filter,
        "n_predictions": len(predictions),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "correlation": correlation,
        "directional_accuracy": directional_accuracy,
        "coverage_50_75": coverage,
    }
    
    metrics_file = REPORTS_DIR / f"backtest_metrics_{position}_{season}_{season_type}.json"
    with metrics_file.open("w") as f:
        json.dump(metrics_dict, f, indent=2)
    console.print(f"[green]Saved metrics to {metrics_file}[/green]")


@app.command()
def generate_predictions(
    season: int = typer.Option(2025, help="Season year"),
    season_type: str = typer.Option("REG", help="Season type: REG or POST"),
    week: int = typer.Option(..., help="Week to generate predictions for"),
    top_n: int = typer.Option(30, help="Number of top players per position"),
) -> None:
    """Run complete pipeline: fetch data, fit models, adjust for defense, and generate predictions.
    
    This is a convenience command that runs all steps in sequence:
    1. Fetch latest data
    2. Fit player models (RB and WR) with advanced algorithm
    3. Calculate defense adjustments (rush and pass)
    4. Generate predictions for the specified week
    """
    _ensure_dirs()
    
    console.print("[bold cyan]Starting complete prediction pipeline...[/bold cyan]")
    console.print(f"Season: {season} {season_type}, Week: {week}, Top N: {top_n}\n")
    
    try:
        # Step 1: Fetch data
        console.print("[bold yellow]Step 1/6: Fetching latest data...[/bold yellow]")
        fetch(season=season, season_type=season_type)
        console.print("[green]✓ Data fetch complete[/green]\n")
        
        # Step 2: Fit RB models
        console.print("[bold yellow]Step 2/6: Fitting RB player models (advanced algorithm)...[/bold yellow]")
        fit_players(
            season=season,
            season_type=season_type,
            position_filter="RB",
            use_weighting=True,
            use_usage_filter=True,
        )
        console.print("[green]✓ RB models fitted[/green]\n")
        
        # Step 3: Fit WR models
        console.print("[bold yellow]Step 3/6: Fitting WR player models (advanced algorithm)...[/bold yellow]")
        fit_players(
            season=season,
            season_type=season_type,
            position_filter="WR",
            use_weighting=True,
            use_usage_filter=True,
        )
        console.print("[green]✓ WR models fitted[/green]\n")
        
        # Step 4: Defense adjustments (RB)
        console.print("[bold yellow]Step 4/6: Calculating rush defense adjustments...[/bold yellow]")
        defense_adjustments(season=season, season_type=season_type, position="RB")
        console.print("[green]✓ Rush defense adjustments calculated[/green]\n")
        
        # Step 5: Defense adjustments (WR)
        console.print("[bold yellow]Step 5/6: Calculating pass defense adjustments...[/bold yellow]")
        defense_adjustments(season=season, season_type=season_type, position="WR")
        console.print("[green]✓ Pass defense adjustments calculated[/green]\n")
        
        # Step 6: Generate predictions
        console.print("[bold yellow]Step 6/6: Generating predictions...[/bold yellow]")
        schedule_predictions(season=season, season_type=season_type, week=week, top_n=top_n)
        
        console.print("\n[bold green]✓ Pipeline complete![/bold green]")
        console.print(f"\nPredictions saved to:")
        console.print(f"  - reports/predictions_RB_{season}_{season_type}_week{week}.csv")
        console.print(f"  - reports/predictions_WR_{season}_{season_type}_week{week}.csv")
        console.print(f"\n[bold]Tip:[/bold] Look for betting lines outside the p25-p75 range for high-confidence plays:")
        console.print(f"  • Line < p25 → Take the OVER (high confidence)")
        console.print(f"  • Line > p75 → Take the UNDER (high confidence)")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed: {e}[/bold red]")
        raise


def run() -> None:
    app()


