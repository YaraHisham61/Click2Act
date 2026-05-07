# UI-Venus-Ground-7B on ScreenSpot-Pro

## Run

```powershell
python src/pipeline/grounder.py experiments/2026-04-27_exp_uivenus_screenspotpro/grounder.yaml
```

## Evaluate

```powershell
python experiments/2026-04-27_exp_uivenus_screenspotpro/run_evaluation.py
```

This evaluator excludes failed/timeout rows from metric calculations and reports:
- evaluated rows
- failed or skipped rows
- point-in-bbox accuracy on evaluated rows only

## Notes

- The model outputs `[x1, y1, x2, y2]`.
- The agent converts this box into a normalized click point by taking the box center.
- Reuse existing evaluator notebooks by pointing to this experiment's `grounder.csv`.
