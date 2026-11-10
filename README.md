# CLakeWQ code repository

**Release: V1.0**  
**Author: Ming Shen · mshen@niglas.ac.cn**

[中文说明](README_zh.md)

This repository contains OLCI scene-level product generation and temporal-statistics code for the 626-lake CLakeWQ study scope. All lakes use the default mask parameters in `01_scene_production/config/mask_parameters.json`. The RF model manifest is V1.0. The model manifest lists the paths and SHA-256 values for the 15 final trained RF bundles.

## Workflows

- `01_scene_production/` generates scene-level Chla, SPM and SDD products and checks lake-year archives.
- `02_spatiotemporal_statistics/` computes daily-first, year-month, annual and trend statistics.
- The workflows use Python 3.11. Install dependencies from each workflow's `requirements.txt`.
- Set external input and output paths in the example configurations before running.

### Scene production

From `01_scene_production`, copy `config/collection_clakewq_626.example.json` to `config/collection_clakewq_626.json`, set the paths, and run:

```powershell
python collection_driver_v2.py --config config/collection_clakewq_626.json --plan
python collection_driver_v2.py --config config/collection_clakewq_626.json --run
python check_collection_products_v2.py --config config/collection_clakewq_626.json --all
```

### Temporal statistics

From `02_spatiotemporal_statistics`, run daily summaries and then year-month summaries:

```powershell
python daily_workflow.py --config config/daily_all_2016_2025.json --local-only
python year_month_runner.py --config config/year_month_all_2016_2025.json --mode plan
python year_month_runner.py --config config/year_month_all_2016_2025.json --mode run
```

The daily-first workflow averages valid scenes per pixel and acquisition date, then calculates monthly climatology, annual and full-period statistics. The year-month workflow summarizes daily values by calendar month. `trend_statistics/trend_statistics.py` calculates Sen slopes and Mann–Kendall p values from annual lake means and writes CSV and JSON tables.

## License

MIT License.

## File appendix

The table lists every file in this repository and its purpose.

| File | Purpose |
|---|---|
| `.gitignore` | Excludes virtual environments, intermediate work and generated products. |
| `01_scene_production/check_collection_products_v2.py` | Checks generated lake-year product archives and metadata. |
| `01_scene_production/collection_driver_v2.py` | Plans and runs lake-year OLCI product production. |
| `01_scene_production/config/collection_clakewq_626.example.json` | Example production settings, 626 lake IDs and external input references. |
| `01_scene_production/config/mask_parameters.json` | Default water, cloud, shadow, bloom and erosion mask parameters. |
| `01_scene_production/config/model_manifest.example.json` | RF V1.0 model metadata, fold IDs, features and expected file hashes. |
| `01_scene_production/pipeline.py` | Processes source scenes, applies masks and RF models, and writes Zarr archives. |
| `01_scene_production/requirements.txt` | Python dependencies for scene production. |
| `01_scene_production/rf_adapter.py` | Loads listed RF bundles and predicts Chla, SPM and SDD. |
| `01_scene_production/runtime.py` | Runtime resource and CPU monitoring helpers. |
| `01_scene_production/science.py` | Scientific processing utilities for input grids, masks and product values. |
| `01_scene_production/vendor/olci_benchmark/__init__.py` | Initializes the bundled OLCI benchmark model package. |
| `01_scene_production/vendor/olci_benchmark/data.py` | Defines OLCI benchmark data and feature utilities required by saved models. |
| `01_scene_production/vendor/olci_benchmark/models.py` | Defines the model-bundle loading and prediction interfaces. |
| `02_spatiotemporal_statistics/collection_pipeline.py` | Builds per-lake annual-product plans and provenance records. |
| `02_spatiotemporal_statistics/config/daily_all_2016_2025.json` | Daily workflow settings for the 2016–2025 collection. |
| `02_spatiotemporal_statistics/config/year_month_all_2016_2025.json` | Year-month workflow settings for the 2016–2025 collection. |
| `02_spatiotemporal_statistics/daily_cached_runner.py` | Runs daily statistics from cached annual product inputs. |
| `02_spatiotemporal_statistics/daily_collection_pipeline.py` | Plans and computes daily-first statistics for the lake collection. |
| `02_spatiotemporal_statistics/daily_parallel_publication.py` | Coordinates parallel preparation of daily-statistics delivery files. |
| `02_spatiotemporal_statistics/daily_publication.py` | Packages daily-statistics outputs and publication metadata. |
| `02_spatiotemporal_statistics/daily_publish_resume.py` | Resumes an interrupted daily-statistics publication step. |
| `02_spatiotemporal_statistics/daily_safe_runner.py` | Runs daily-statistics tasks with checkpoint and retry handling. |
| `02_spatiotemporal_statistics/daily_statistics_core.py` | Computes daily-first pixel and lake temporal summaries. |
| `02_spatiotemporal_statistics/daily_status.py` | Reports daily-statistics workflow progress and status. |
| `02_spatiotemporal_statistics/daily_workflow.py` | Entry point for configuring and running the daily workflow. |
| `02_spatiotemporal_statistics/pipeline.py` | Computes pooled temporal summaries from scene or daily inputs. |
| `02_spatiotemporal_statistics/requirements.txt` | Python dependencies for temporal statistics. |
| `02_spatiotemporal_statistics/statistics_core.py` | Shared aggregation and quality-control functions for statistics. |
| `02_spatiotemporal_statistics/trend_statistics/requirements.txt` | Python dependencies for trend-table generation. |
| `02_spatiotemporal_statistics/trend_statistics/trend_statistics.py` | Calculates Sen slopes and Mann–Kendall statistics and writes tables. |
| `02_spatiotemporal_statistics/verify_collection_statistics.py` | Checks collection-level temporal-statistics inputs and outputs. |
| `02_spatiotemporal_statistics/verify_daily_statistics.py` | Checks daily-first statistics and representative source pixels. |
| `02_spatiotemporal_statistics/verify_statistics.py` | Checks temporal-statistics outputs against configured rules. |
| `02_spatiotemporal_statistics/year_month_engine.py` | Computes year-month lake and pixel summaries from daily values. |
| `02_spatiotemporal_statistics/year_month_runner.py` | Plans and runs year-month statistics for selected lakes. |
| `LICENSE` | MIT license text. |
| `MANIFEST.csv` | File inventory with sizes and SHA-256 checksums. |
| `README.md` | English repository guide and complete file appendix. |
| `README_zh.md` | Chinese repository guide and complete file appendix. |
