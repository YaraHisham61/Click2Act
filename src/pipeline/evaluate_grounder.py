import ast
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def _parse_annotation(annotation: str) -> dict:
    if isinstance(annotation, dict):
        return annotation
    return ast.literal_eval(annotation)


def _point_in_bbox(row: pd.Series) -> bool:
    w, h = row["img_size"]
    px = float(row["coord_x"]) * float(w)
    py = float(row["coord_y"]) * float(h)
    x1, y1, x2, y2 = row["bbox"]
    return float(x1) <= px <= float(x2) and float(y1) <= py <= float(y2)


def main(config: dict) -> None:
    input_csv = Path(config["input_csv"])
    output_summary_json = Path(config["output_summary_json"])
    output_valid_csv = Path(config.get("output_valid_csv", input_csv.with_name("grounder_valid_only.csv")))

    df = pd.read_csv(input_csv)

    parsed = df["annotation"].apply(_parse_annotation)
    df["bbox"] = parsed.apply(lambda x: x["bbox"])
    df["img_size"] = parsed.apply(lambda x: x["img_size"])
    df["ui_type"] = parsed.apply(lambda x: x.get("ui_type"))
    df["application"] = parsed.apply(lambda x: x.get("application"))
    df["platform"] = parsed.apply(lambda x: x.get("platform"))

    has_status = "status" in df.columns
    if has_status:
        valid_mask = (df["status"] == "ok") & df["coord_x"].notna() & df["coord_y"].notna()
    else:
        valid_mask = df["coord_x"].notna() & df["coord_y"].notna()

    valid_df = df[valid_mask].copy()
    valid_df["point_in_bbox"] = valid_df.apply(_point_in_bbox, axis=1)

    total = int(len(df))
    valid = int(len(valid_df))
    failed = int(total - valid)

    summary = {
        "total_rows": total,
        "evaluated_rows": valid,
        "failed_or_skipped_rows": failed,
        "point_in_bbox_accuracy": float(valid_df["point_in_bbox"].mean()) if valid > 0 else None,
    }

    by_platform = (
        valid_df.groupby("platform")["point_in_bbox"].mean().sort_values(ascending=False).to_dict()
        if valid > 0
        else {}
    )
    by_ui_type = (
        valid_df.groupby("ui_type")["point_in_bbox"].mean().sort_values(ascending=False).to_dict()
        if valid > 0
        else {}
    )
    summary["accuracy_by_platform"] = {k: float(v) for k, v in by_platform.items()}
    summary["accuracy_by_ui_type"] = {k: float(v) for k, v in by_ui_type.items()}

    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_valid_csv.parent.mkdir(parents=True, exist_ok=True)

    output_valid_columns = [
        c
        for c in [
            "idx",
            "task",
            "coord_x",
            "coord_y",
            "status",
            "failed_reason",
            "latency_s",
            "platform",
            "application",
            "ui_type",
            "point_in_bbox",
            "bbox",
            "img_size",
            "annotation",
        ]
        if c in valid_df.columns
    ]

    valid_df[output_valid_columns].to_csv(output_valid_csv, index=False)
    output_summary_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())
    main(config)
