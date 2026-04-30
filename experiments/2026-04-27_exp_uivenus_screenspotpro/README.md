# UI-Venus-Ground-7B on ScreenSpot-Pro

## Run

```powershell
python src/pipeline/grounder.py experiments/2026-04-27_exp_uivenus_screenspotpro/grounder.yaml
```

## Notes

- The model outputs `[x1, y1, x2, y2]`.
- The agent converts this box into a normalized click point by taking the box center.
- Reuse existing evaluator notebooks by pointing to this experiment's `grounder.csv`.
