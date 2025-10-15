#!/usr/bin/env python3
"""
Validation script to compare old vs new prediction models.

This script runs backtests for both the baseline (old) and improved (new) models
and compares their performance on weeks 5-6 after training on weeks 1-4.
"""

import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT / "line_predictions")
    return result.returncode, result.stdout, result.stderr


def run_backtest(position: str, use_weighting: bool, use_usage_filter: bool, label: str) -> dict:
    """Run a backtest with specific settings"""
    print(f"\n{'=' * 70}")
    print(f"Running backtest: {label} ({position})")
    print(f"{'=' * 70}")
    
    cmd = [
        "uv", "run", "line-predictions", "backtest",
        "--season", "2025",
        "--season-type", "REG",
        "--train-weeks-str", "1,2,3,4",
        "--test-weeks-str", "5,6",
        "--position", position,
    ]
    
    if use_weighting:
        cmd.append("--use-weighting")
    else:
        cmd.extend(["--no-use-weighting"])
    
    if use_usage_filter:
        cmd.append("--use-usage-filter")
    else:
        cmd.extend(["--no-use-usage-filter"])
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode != 0:
        print(f"ERROR: Backtest failed for {label}")
        print(f"STDERR: {stderr}")
        return None
    
    print(stdout)
    
    # Parse metrics from JSON file
    metrics_file = REPORTS_DIR / f"backtest_metrics_{position}_2025_REG.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        metrics['label'] = label
        return metrics
    else:
        print(f"WARNING: Metrics file not found at {metrics_file}")
        return None


def create_comparison_plot(results: list[dict], output_path: Path):
    """Create comparison plots for baseline vs improved models"""
    df = pd.DataFrame(results)
    
    # Filter to one position for cleaner plot (can be modified)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison: Baseline vs Improved', fontsize=16, fontweight='bold')
    
    metrics = ['mae', 'rmse', 'correlation', 'directional_accuracy', 'mape', 'coverage_50_75']
    titles = ['MAE (yards)', 'RMSE (yards)', 'Correlation', 'Directional Accuracy (%)', 'MAPE (%)', 'Coverage 50-75th (%)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        
        # Group by position and label
        for position in df['position'].unique():
            pos_data = df[df['position'] == position]
            
            baseline = pos_data[pos_data['label'].str.contains('Baseline')]
            improved = pos_data[pos_data['label'].str.contains('Improved')]
            
            if not baseline.empty and not improved.empty:
                x = [f"{position}\nBaseline", f"{position}\nImproved"]
                y = [baseline[metric].values[0], improved[metric].values[0]]
                
                colors = ['#ff7f0e' if position == 'RB' else '#2ca02c' for _ in y]
                bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line for target values where applicable
        if metric == 'directional_accuracy':
            ax.axhline(y=50, color='red', linestyle='--', label='Random (50%)', alpha=0.5)
            ax.legend()
        elif metric == 'coverage_50_75':
            ax.axhline(y=25, color='red', linestyle='--', label='Target (25%)', alpha=0.5)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")


def generate_comparison_report(results: list[dict], output_path: Path):
    """Generate a markdown comparison report"""
    report = []
    report.append("# Model Validation Report: Baseline vs Improved")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append("This report compares the performance of the baseline model (equal weighting, no filtering)")
    report.append("against the improved model (recency weighting, usage filtering, position-specific tuning).")
    report.append("")
    report.append("**Training Data:** Weeks 1-4")
    report.append("**Test Data:** Weeks 5-6")
    report.append("")
    
    # Group results by position
    df = pd.DataFrame(results)
    
    for position in ['RB', 'WR']:
        report.append(f"## {position} Position Results")
        report.append("")
        
        pos_results = df[df['position'] == position]
        baseline = pos_results[pos_results['label'].str.contains('Baseline')].iloc[0] if not pos_results[pos_results['label'].str.contains('Baseline')].empty else None
        improved = pos_results[pos_results['label'].str.contains('Improved')].iloc[0] if not pos_results[pos_results['label'].str.contains('Improved')].empty else None
        
        if baseline is not None and improved is not None:
            report.append("| Metric | Baseline | Improved | Change | % Change |")
            report.append("|--------|----------|----------|--------|----------|")
            
            metrics = [
                ('MAE (yards)', 'mae', 'lower'),
                ('RMSE (yards)', 'rmse', 'lower'),
                ('MAPE (%)', 'mape', 'lower'),
                ('Correlation', 'correlation', 'higher'),
                ('Directional Accuracy (%)', 'directional_accuracy', 'higher'),
                ('Coverage 50-75th (%)', 'coverage_50_75', 'closer_to_25'),
            ]
            
            for label, key, direction in metrics:
                base_val = baseline[key]
                imp_val = improved[key]
                change = imp_val - base_val
                
                if direction == 'closer_to_25':
                    pct_change = ((abs(25 - imp_val) - abs(25 - base_val)) / abs(25 - base_val)) * 100
                    improvement = "✅" if abs(25 - imp_val) < abs(25 - base_val) else "❌"
                elif direction == 'lower':
                    pct_change = (change / base_val) * 100 if base_val != 0 else 0
                    improvement = "✅" if change < 0 else "❌"
                else:  # higher
                    pct_change = (change / base_val) * 100 if base_val != 0 else 0
                    improvement = "✅" if change > 0 else "❌"
                
                report.append(f"| {label} | {base_val:.2f} | {imp_val:.2f} | {change:+.2f} | {pct_change:+.1f}% {improvement} |")
            
            report.append("")
            report.append(f"**Predictions Generated:** {int(improved['n_predictions'])}")
            report.append("")
            
            # Key findings
            report.append("### Key Findings")
            report.append("")
            
            mae_improvement = ((baseline['mae'] - improved['mae']) / baseline['mae']) * 100
            corr_improvement = ((improved['correlation'] - baseline['correlation']) / abs(baseline['correlation'])) * 100 if baseline['correlation'] != 0 else 0
            
            if mae_improvement > 0:
                report.append(f"- ✅ MAE improved by {mae_improvement:.1f}% ({baseline['mae']:.1f} → {improved['mae']:.1f} yards)")
            else:
                report.append(f"- ❌ MAE worsened by {abs(mae_improvement):.1f}% ({baseline['mae']:.1f} → {improved['mae']:.1f} yards)")
            
            if improved['correlation'] > baseline['correlation']:
                report.append(f"- ✅ Correlation improved by {corr_improvement:.1f}% ({baseline['correlation']:.3f} → {improved['correlation']:.3f})")
            else:
                report.append(f"- ❌ Correlation worsened ({baseline['correlation']:.3f} → {improved['correlation']:.3f})")
            
            coverage_improvement = abs(25 - improved['coverage_50_75']) < abs(25 - baseline['coverage_50_75'])
            if coverage_improvement:
                report.append(f"- ✅ Coverage closer to target 25% ({baseline['coverage_50_75']:.1f}% → {improved['coverage_50_75']:.1f}%)")
            else:
                report.append(f"- ❌ Coverage further from target ({baseline['coverage_50_75']:.1f}% → {improved['coverage_50_75']:.1f}%)")
            
            report.append("")
    
    report.append("## Conclusion")
    report.append("")
    report.append("### Improvements Summary")
    report.append("")
    report.append("The improved model includes:")
    report.append("- **Recency weighting**: Exponential decay favoring recent games (decay=0.90 for RB, 0.85 for WR)")
    report.append("- **Usage filtering**: Minimum snap % and touch/target share thresholds")
    report.append("- **Position-specific sigma**: Tighter for RBs (0.90x), wider for WRs (1.15x)")
    report.append("- **Calibrated intervals**: Inflated sigma for better coverage (1.3x for RB, 1.5x for WR)")
    report.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nSaved comparison report to {output_path}")


def main():
    """Main validation workflow"""
    print("=" * 70)
    print("MODEL VALIDATION: Baseline vs Improved")
    print("=" * 70)
    print("")
    print("This script will:")
    print("1. Run baseline model (no improvements)")
    print("2. Run improved model (with all enhancements)")
    print("3. Compare results and generate report")
    print("")
    
    results = []
    
    # Run backtests for both positions with both models
    for position in ['RB', 'WR']:
        # Baseline model
        baseline_result = run_backtest(
            position=position,
            use_weighting=False,
            use_usage_filter=False,
            label=f"Baseline ({position})"
        )
        if baseline_result:
            results.append(baseline_result)
        
        # Improved model
        improved_result = run_backtest(
            position=position,
            use_weighting=True,
            use_usage_filter=True,
            label=f"Improved ({position})"
        )
        if improved_result:
            results.append(improved_result)
    
    if not results:
        print("\nERROR: No results collected. Check that backtests ran successfully.")
        sys.exit(1)
    
    # Generate comparison outputs
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON REPORT")
    print("=" * 70)
    
    output_plot = REPORTS_DIR / "model_comparison.png"
    create_comparison_plot(results, output_plot)
    
    output_report = REPORTS_DIR / "MODEL_COMPARISON.md"
    generate_comparison_report(results, output_report)
    
    # Save results as JSON
    results_json = REPORTS_DIR / "validation_results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {results_json}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Comparison plot: {output_plot}")
    print(f"  - Comparison report: {output_report}")
    print(f"  - Raw results: {results_json}")
    print("")


if __name__ == "__main__":
    main()

