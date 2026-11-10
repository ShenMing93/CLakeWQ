from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
from scipy.stats import norm

PARAMETERS = ("Chla", "SPM", "SDD")
REGIONS = ("EPL", "YGPL", "IMXL", "NPML", "TPL")
AREA_BINS = (0, 30, 50, 100, 500, np.inf)
AREA_CLASSES = ("<30", "30–<50", "50–<100", "100–<500", "≥500")
MIN_YEARS = 2


def mk_sen(years: np.ndarray, values: np.ndarray) -> dict[str, object]:
    pairs = sorted(
        (int(year), float(value))
        for year, value in zip(years, values)
        if pd.notna(year) and pd.notna(value) and np.isfinite(float(value))
    )
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    n = len(y)
    result: dict[str, object] = {
        "n_years": n,
        "years_used": ",".join(map(str, x.astype(int))),
        "sen_slope_year": np.nan,
        "sen_slope_decade": np.nan,
        "mk_statistic": np.nan,
        "mk_z": np.nan,
        "mk_p": np.nan,
    }
    if n < MIN_YEARS:
        return result

    score = 0
    slopes: list[float] = []
    for i in range(n - 1):
        differences = y[i + 1 :] - y[i]
        score += int(np.sign(differences).sum())
        slopes.extend((differences / (x[i + 1 :] - x[i])).tolist())

    _, ties = np.unique(y, return_counts=True)
    variance = (
        n * (n - 1) * (2 * n + 5)
        - np.sum(ties * (ties - 1) * (2 * ties + 5))
    ) / 18
    z_value = (score - np.sign(score)) / np.sqrt(variance) if variance > 0 else 0.0
    slope = float(np.median(slopes))
    result.update(
        sen_slope_year=slope,
        sen_slope_decade=slope * 10,
        mk_statistic=int(score),
        mk_z=float(z_value),
        mk_p=float(2 * norm.sf(abs(z_value))),
    )
    return result


def load_inputs(annual_path: Path, lakes_path: Path, layer: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake_fields = ["lake_id", "name_cn", "region", "area_km2", "lon", "lat"]
    lakes = pyogrio.read_dataframe(lakes_path, layer=layer, read_geometry=False)[lake_fields]
    lakes = lakes.rename(
        columns={"area_km2": "lake_area_km2", "lon": "longitude", "lat": "latitude"}
    )
    annual = pyogrio.read_dataframe(annual_path, read_geometry=False)
    required = {"lake_id", "year", *PARAMETERS, "region"}
    missing = required.difference(annual.columns)
    if missing:
        raise ValueError(f"Annual input is missing fields: {', '.join(sorted(missing))}")

    lakes["lake_id"] = lakes.lake_id.astype(str)
    annual["lake_id"] = annual.lake_id.astype(str)
    annual["year"] = pd.to_numeric(annual.year, errors="coerce")
    annual = annual[
        annual.lake_id.isin(set(lakes.lake_id))
        & annual.year.between(2016, 2025)
    ].copy()
    annual["year"] = annual.year.astype(int)
    annual["region"] = annual.region.replace({"NPLM": "NPML", "NMPL": "NPML"})

    if lakes.lake_id.nunique() != len(lakes):
        raise ValueError("Lake identifiers in the study-area layer are not unique")
    if annual.duplicated(["lake_id", "year"]).any():
        raise ValueError("Annual input contains duplicate lake-year rows")
    return annual, lakes


def write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding="utf-8-sig")


def build_trend_tables(annual: pd.DataFrame, lakes: pd.DataFrame, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    write_csv(annual, output / "annual_lake_means_2016_2025.csv")
    metadata = ["lake_id", "name_cn", "region", "lake_area_km2", "longitude", "latitude"]
    lake_area = lakes[["lake_id", "lake_area_km2"]].copy()
    lake_area["lake_area_km2"] = pd.to_numeric(lake_area.lake_area_km2, errors="coerce")
    lake_area["area_class"] = pd.cut(
        lake_area.lake_area_km2, bins=AREA_BINS, right=False, labels=AREA_CLASSES
    )
    if lake_area.area_class.isna().any():
        raise ValueError("Some target lakes do not have a usable area class")

    annual = annual.copy()
    annual_area = annual.merge(
        lake_area[["lake_id", "area_class"]], on="lake_id", how="left", validate="many_to_one"
    )
    trend_tables: list[pd.DataFrame] = []
    eligible_ids: dict[str, set[str]] = {}

    for parameter in PARAMETERS:
        annual[parameter] = pd.to_numeric(annual[parameter], errors="coerce")
        trends = pd.DataFrame(
            [
                {"lake_id": lake_id, **mk_sen(group.year.to_numpy(), group[parameter].to_numpy())}
                for lake_id, group in annual.groupby("lake_id", sort=False)
            ]
        )
        table = lakes[metadata].merge(trends, on="lake_id", how="left", validate="one_to_one")
        table.insert(len(metadata), "parameter", parameter)
        table["sen_slope_year"] = pd.to_numeric(table.sen_slope_year, errors="coerce")
        table["mk_p"] = pd.to_numeric(table.mk_p, errors="coerce")
        eligible_ids[parameter] = set(table.loc[table.sen_slope_year.notna(), "lake_id"])
        write_csv(table, output / f"lake_{parameter.lower()}_mean_mk_sen_2016_2025.csv")
        trend_tables.append(table)

    all_trends = pd.concat(trend_tables, ignore_index=True)
    write_csv(all_trends, output / "lake_wq_mean_mk_sen_2016_2025.csv")

    grouped_rows: list[dict[str, object]] = []
    for parameter in PARAMETERS:
        eligible_annual = annual[annual.lake_id.isin(eligible_ids[parameter])]
        for (region, year), group in eligible_annual.groupby(["region", "year"], dropna=True):
            values = pd.to_numeric(group[parameter], errors="coerce").dropna()
            grouped_rows.append(
                {"parameter": parameter, "region": region, "year": int(year),
                 "mean_lake_value": float(values.mean()) if len(values) else np.nan,
                 "n_lakes": int(len(values))}
            )
        for year, group in eligible_annual.groupby("year", dropna=True):
            values = pd.to_numeric(group[parameter], errors="coerce").dropna()
            grouped_rows.append(
                {"parameter": parameter, "region": "China", "year": int(year),
                 "mean_lake_value": float(values.mean()) if len(values) else np.nan,
                 "n_lakes": int(len(values))}
            )
    regional_annual = pd.DataFrame(grouped_rows)
    write_csv(regional_annual, output / "regional_annual_lake_mean_2016_2025.csv")
    regional_trends = []
    for parameter in PARAMETERS:
        for region in ("China", *REGIONS):
            group = regional_annual[
                (regional_annual.parameter == parameter)
                & (regional_annual.region == region)
            ].sort_values("year")
            regional_trends.append(
                {"parameter": parameter, "region": region,
                 **mk_sen(group.year.to_numpy(), group.mean_lake_value.to_numpy())}
            )
    write_csv(
        pd.DataFrame(regional_trends),
        output / "regional_mk_sen_2016_2025.csv",
    )

    years = sorted(annual.year.dropna().unique().astype(int))
    area_rows: list[dict[str, object]] = []
    area_trend_rows: list[dict[str, object]] = []
    for parameter in PARAMETERS:
        eligible_area = annual_area[annual_area.lake_id.isin(eligible_ids[parameter])]
        for area_class in AREA_CLASSES:
            series = []
            for year in years:
                group = eligible_area[
                    (eligible_area.area_class == area_class) & (eligible_area.year == year)
                ]
                values = pd.to_numeric(group[parameter], errors="coerce").dropna()
                mean_value = float(values.mean()) if len(values) else np.nan
                row = {"parameter": parameter, "area_class": area_class, "year": year,
                       "mean_lake_value": mean_value, "n_lakes": int(len(values))}
                area_rows.append(row)
                series.append(row)
            area_trend_rows.append(
                {"parameter": parameter, "area_class": area_class,
                 **mk_sen(
                     np.asarray([row["year"] for row in series]),
                     np.asarray([row["mean_lake_value"] for row in series]),
                 )}
            )

    write_csv(pd.DataFrame(area_rows), output / "area_class_annual_lake_mean_2016_2025.csv")
    write_csv(pd.DataFrame(area_trend_rows), output / "area_class_mk_sen_2016_2025.csv")

    return {
        "period": "2016–2025",
        "minimum_valid_years": MIN_YEARS,
        "parameters": list(PARAMETERS),
        "lake_trend_counts": {
            parameter: int(len(ids)) for parameter, ids in eligible_ids.items()
        },
        "trend_method": (
            "Sen median of pairwise slopes using actual year differences; "
            "two-sided Mann–Kendall normal approximation with tie variance "
            "and continuity correction."
        ),
        "outputs": [path.name for path in sorted(output.glob("*.csv"))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate lake, region and area-class Sen–MK trends from annual lake means."
    )
    parser.add_argument("--annual", type=Path, required=True, help="Annual lake-mean vector file")
    parser.add_argument("--lakes", type=Path, required=True, help="Study-lake GeoPackage")
    parser.add_argument("--layer", default="study_lakes_626", help="Lake layer in the GeoPackage")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for tables")
    args = parser.parse_args()

    annual, lakes = load_inputs(args.annual, args.lakes, args.layer)
    summary = build_trend_tables(annual, lakes, args.output)
    (args.output / "trend_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
