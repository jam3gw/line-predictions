# Prediction Evaluation Report: Weeks 5 & 6

## Executive Summary

This report evaluates the performance of our rushing and receiving yard predictions for NFL weeks 5 and 6. The predictions use a log-normal distribution model with defensive adjustments to forecast player performance.

---

## Overall Performance

### Running Backs (RB)

**Combined Weeks 5 & 6 Metrics:**
- **Mean Absolute Error (MAE)**: 38.6 yards
- **Root Mean Square Error (RMSE)**: 45.5 yards
- **Mean Absolute Percentage Error (MAPE)**: 129.4%
- **Correlation**: 0.174
- **Directional Accuracy**: 53.1% (slightly better than random)
- **Coverage (Median to P75)**: 16.3% (target: ~25%)

**Key Observations:**
- Week 6 performance (MAE: 41.9 yards) was notably worse than Week 5 (MAE: 35.4 yards)
- The correlation is weak but positive (0.17), indicating some predictive signal
- Directional accuracy of 53.1% shows we're only marginally better than a coin flip at predicting above/below median performance
- Coverage of 16.3% is below the expected ~25% for a proper 50th-75th percentile interval

### Wide Receivers (WR)

**Combined Weeks 5 & 6 Metrics:**
- **Mean Absolute Error (MAE)**: 44.2 yards
- **Root Mean Square Error (RMSE)**: 52.4 yards
- **Mean Absolute Percentage Error (MAPE)**: 122.6%
- **Correlation**: -0.057 (slight negative correlation)
- **Directional Accuracy**: 45.5% (worse than random)
- **Coverage (Median to P75)**: 13.6% (target: ~25%)

**Key Observations:**
- Wide receiver predictions performed worse than running back predictions
- Week 6 (MAE: 53.7 yards) showed significant degradation from Week 5 (MAE: 34.7 yards)
- The slight negative correlation (-0.057) suggests the model is essentially uncorrelated with actual performance
- Directional accuracy of 45.5% is worse than random chance
- Coverage of 13.6% indicates prediction intervals are too narrow or poorly calibrated

---

## Position-Specific Analysis

### Running Backs

#### Week 5 Performance
- **MAE**: 35.4 yards
- **Correlation**: 0.11 (very weak)
- **Best Predictions**: Christian McCaffrey (1.4 yards error), Ashton Jeanty (1.6 yards error)
- **Biggest Misses**: 
  - Travis Etienne: Predicted 120.0 yards, Actual 49.0 yards (-71.0)
  - Jahmyr Gibbs: Predicted 112.3 yards, Actual 54.0 yards (-58.3)

#### Week 6 Performance
- **MAE**: 41.9 yards (worse than Week 5)
- **Correlation**: 0.21 (slightly improved)
- **Best Predictions**: Breece Hall (2.8 yards error), Kenneth Walker III (2.8 yards error)
- **Biggest Misses**:
  - Jeremy McNichols: Predicted 101.4 yards, Actual 5.0 yards (-96.4)
  - Quinshon Judkins: Predicted 132.0 yards, Actual 36.0 yards (-96.0)

#### Top Overperformers (Beat Predictions)
1. **De'Von Achane** (Week 6): 128.0 actual vs 49.1 predicted (+78.9)
2. **Jacory Croskey-Merritt** (Week 5): 111.0 actual vs 43.6 predicted (+67.4)
3. **Javonte Williams** (Week 5): 135.0 actual vs 76.2 predicted (+58.8)
4. **Kenneth Walker III** (Week 5): 86.0 actual vs 28.1 predicted (+57.9)
5. **Breece Hall** (Week 5): 113.0 actual vs 56.9 predicted (+56.1)

#### Top Underperformers (Missed Predictions)
1. **Jeremy McNichols** (Week 6): 5.0 actual vs 101.4 predicted (-96.4)
2. **Quinshon Judkins** (Week 6): 36.0 actual vs 132.0 predicted (-96.0)
3. **Travis Etienne** (Week 5): 49.0 actual vs 120.0 predicted (-71.0)
4. **J.K. Dobbins** (Week 6): 40.0 actual vs 110.5 predicted (-70.5)
5. **Travis Etienne** (Week 6): 27.0 actual vs 87.5 predicted (-60.5)

### Wide Receivers

#### Week 5 Performance
- **MAE**: 34.7 yards
- **Correlation**: 0.13 (very weak but positive)
- **Best Predictions**: Tetairoa McMillan (1.7 yards error), Tutu Atwell (13.0 yards error)
- **Biggest Misses**:
  - Jameson Williams: Predicted 80.6 yards, Actual 9.0 yards (-71.6)
  - Quentin Johnston: Predicted 92.0 yards, Actual 40.0 yards (-52.0)

#### Week 6 Performance
- **MAE**: 53.7 yards (significant degradation)
- **Correlation**: -0.14 (negative correlation!)
- **Best Predictions**: Jameson Williams (8.4 yards error), Ja'Marr Chase (9.9 yards error)
- **Biggest Misses**:
  - Puka Nacua: Predicted 175.9 yards, Actual 28.0 yards (-147.9)
  - Tetairoa McMillan: Predicted 100.0 yards, Actual 29.0 yards (-71.0)

#### Top Overperformers (Beat Predictions)
1. **Drake London** (Week 6): 158.0 actual vs 42.2 predicted (+115.8)
2. **George Pickens** (Week 6): 168.0 actual vs 71.3 predicted (+96.7)
3. **Emeka Egbuka** (Week 5): 163.0 actual vs 79.5 predicted (+83.5)
4. **Justin Jefferson** (Week 5): 123.0 actual vs 61.9 predicted (+61.1)
5. **Jaxon Smith-Njigba** (Week 6): 162.0 actual vs 108.0 predicted (+54.0)

#### Top Underperformers (Missed Predictions)
1. **Puka Nacua** (Week 6): 28.0 actual vs 175.9 predicted (-147.9)
2. **Jameson Williams** (Week 5): 9.0 actual vs 80.6 predicted (-71.6)
3. **Tetairoa McMillan** (Week 6): 29.0 actual vs 100.0 predicted (-71.0)
4. **Michael Pittman** (Week 6): 20.0 actual vs 82.5 predicted (-62.5)
5. **Zay Flowers** (Week 6): 46.0 actual vs 106.5 predicted (-60.5)

---

## Key Findings

### Strengths
1. **Some predictive signal for RBs**: The positive correlation (0.17) for running backs suggests the model captures some meaningful patterns
2. **Best case accuracy**: When the model is correct, errors can be as low as 1-2 yards
3. **Consistent player identification**: Successfully identified top performers like James Cook, Saquon Barkley, and Ja'Marr Chase

### Weaknesses
1. **Week-to-week volatility**: Performance significantly degraded from Week 5 to Week 6 for both positions
2. **Poor WR predictions**: Wide receiver predictions showed negative correlation in Week 6
3. **Large errors**: MAPE over 120% indicates predictions can be off by more than the actual value
4. **Overconfidence**: Prediction intervals (median to p75) are too narrow, covering only 13-16% of outcomes vs expected 25%
5. **Limited sample size**: Only 22-25 matched players per week limits statistical confidence
6. **Injury/usage risk**: Major misses often involve unexpected low usage (e.g., Jeremy McNichols, Puka Nacua)

### Critical Issues
1. **No better than chance for WRs**: 45.5% directional accuracy is worse than random
2. **Massive outliers**: Some predictions were off by 90+ yards, suggesting missing context (injuries, game script, etc.)
3. **Model degradation**: Performance worsened in Week 6, possibly due to:
   - Changing team strategies
   - Injuries not reflected in historical data
   - Small sample size amplifying noise

---

## Recommendations

### Immediate Improvements
1. **Injury data integration**: Incorporate real-time injury reports and practice participation
2. **Game script considerations**: Add projected game flow (pass/run heavy based on Vegas lines)
3. **Usage patterns**: Weight recent weeks more heavily to capture role changes
4. **Wider prediction intervals**: Increase uncertainty bounds to improve coverage
5. **Minimum snap/touch thresholds**: Filter out players with uncertain usage

### Medium-Term Enhancements
1. **Ensemble approach**: Combine with other models (Vegas props, DFS pricing)
2. **Position-specific features**: Different features for RB vs WR (target share, red zone usage)
3. **Opponent-specific adjustments**: More granular defensive metrics (slot coverage, run defense by gap)
4. **Weather integration**: Add weather conditions for outdoor games
5. **Home/away splits**: Account for venue-specific performance

### Long-Term Strategy
1. **Expand sample size**: Need more weeks of data to validate model stability
2. **Build confidence intervals**: Use bootstrapping or Bayesian methods for uncertainty quantification
3. **Benchmark against Vegas**: Compare predictions to over/under betting lines
4. **Track prediction drift**: Monitor how model performance changes over time
5. **Consider ensemble meta-model**: Use multiple prediction approaches and weight by recent performance

---

## Visualizations

The following plots have been generated in `reports/plots/`:
- `evaluation_RB_week5.png`: RB predictions vs actuals for Week 5
- `evaluation_RB_week6.png`: RB predictions vs actuals for Week 6
- `evaluation_WR_week5.png`: WR predictions vs actuals for Week 5
- `evaluation_WR_week6.png`: WR predictions vs actuals for Week 6

Each plot includes:
1. **Median predictions scatter**: Shows predicted median vs actual yards
2. **Expected value scatter**: Shows predicted expected value vs actual yards
3. **Residual plot**: Shows prediction errors relative to predicted values

---

## Conclusion

The current prediction model shows **limited but non-zero predictive value for running backs** (correlation: 0.17) but **essentially no predictive value for wide receivers** (correlation: -0.06). The large errors (MAE: 38-44 yards) and poor calibration (coverage: 13-16% vs target 25%) indicate the model needs significant improvements before it can be reliably used for decision-making.

The most promising path forward is to:
1. Integrate contextual data (injuries, game script, usage)
2. Improve prediction intervals for proper uncertainty quantification
3. Develop position-specific models rather than a one-size-fits-all approach
4. Validate against industry benchmarks (Vegas lines, expert projections)

**Current Model Grade: C-**
- RB predictions: C+ (some signal, but high error)
- WR predictions: D (no reliable signal)

