# CLakeWQ代码仓库

**版本：V1.0**  
**作者：Ming Shen · mshen@niglas.ac.cn**

[English documentation](README.md)

本仓库包含CLakeWQ研究范围内626个湖泊的OLCI逐景产品生产和时间统计代码。所有湖泊统一使用 `01_scene_production/config/mask_parameters.json` 中的默认掩膜参数。RF模型清单版本为V1.0。模型清单列出15个最终训练RF模型包的路径和SHA-256值。本版本不含RF权重；将模型包放到清单所列路径后即可运行模型推理。不含训练数据和源影像。

## 流程

- `01_scene_production/`：生成Chla、SPM和SDD逐景产品，并核验湖泊年度归档。
- `02_spatiotemporal_statistics/`：计算日优先、年月、年度和趋势统计。
- 运行环境为Python 3.11。分别根据各流程目录中的 `requirements.txt` 安装依赖。
- 运行前在配置示例中设置外部输入和输出路径。

### 逐景产品生产

在 `01_scene_production` 目录，将 `config/collection_clakewq_626.example.json` 复制为 `config/collection_clakewq_626.json` 并填写路径，然后运行：

```powershell
python collection_driver_v2.py --config config/collection_clakewq_626.json --plan
python collection_driver_v2.py --config config/collection_clakewq_626.json --run
python check_collection_products_v2.py --config config/collection_clakewq_626.json --all
```

### 时间统计

在 `02_spatiotemporal_statistics` 目录先运行日统计，再运行年月统计：

```powershell
python daily_workflow.py --config config/daily_all_2016_2025.json --local-only
python year_month_runner.py --config config/year_month_all_2016_2025.json --mode plan
python year_month_runner.py --config config/year_month_all_2016_2025.json --mode run
```

日优先流程先对每个像元同一采集日内的有效景值求平均，再计算月气候态、年度和全期统计。年月流程按日历月汇总日值。`trend_statistics/trend_statistics.py`基于湖泊年度均值计算Sen斜率和Mann–Kendall p值，并写出CSV和JSON表。

## 许可

MIT许可证。

## 文件附录

下表列出仓库中的每个文件及其用途。

| 文件 | 用途 |
|---|---|
| `.gitignore` | 忽略虚拟环境、中间文件和生成产品。 |
| `01_scene_production/check_collection_products_v2.py` | 核验湖泊年度产品归档及元数据。 |
| `01_scene_production/collection_driver_v2.py` | 规划并运行湖泊年度OLCI产品生产。 |
| `01_scene_production/config/collection_clakewq_626.example.json` | 逐景生产配置示例、626个湖泊ID及外部输入路径。 |
| `01_scene_production/config/mask_parameters.json` | 水体、云、阴影、藻华和腐蚀掩膜默认参数。 |
| `01_scene_production/config/model_manifest.example.json` | RF V1.0模型元数据、折编号、特征、模型路径及SHA-256。 |
| `01_scene_production/pipeline.py` | 处理源影像，应用掩膜和RF模型，并写出Zarr归档。 |
| `01_scene_production/requirements.txt` | 逐景生产依赖包列表。 |
| `01_scene_production/rf_adapter.py` | 加载清单中的RF模型并预测Chla、SPM和SDD。 |
| `01_scene_production/runtime.py` | 运行资源与CPU监控函数。 |
| `01_scene_production/science.py` | 输入网格、掩膜及产品值的科学处理函数。 |
| `01_scene_production/vendor/olci_benchmark/__init__.py` | 初始化随代码提供的OLCI模型运行包。 |
| `01_scene_production/vendor/olci_benchmark/data.py` | 提供模型调用的OLCI数据与特征函数。 |
| `01_scene_production/vendor/olci_benchmark/models.py` | 定义模型包加载和预测接口。 |
| `02_spatiotemporal_statistics/collection_pipeline.py` | 构建逐湖年度产品任务及来源记录。 |
| `02_spatiotemporal_statistics/config/daily_all_2016_2025.json` | 2016–2025日统计流程配置。 |
| `02_spatiotemporal_statistics/config/year_month_all_2016_2025.json` | 2016–2025年月统计流程配置。 |
| `02_spatiotemporal_statistics/daily_cached_runner.py` | 使用缓存的年度产品输入运行日统计。 |
| `02_spatiotemporal_statistics/daily_collection_pipeline.py` | 规划并计算湖泊集合的日优先统计。 |
| `02_spatiotemporal_statistics/daily_parallel_publication.py` | 并行准备日统计交付文件。 |
| `02_spatiotemporal_statistics/daily_publication.py` | 打包日统计结果及发布元数据。 |
| `02_spatiotemporal_statistics/daily_publish_resume.py` | 恢复中断的日统计发布步骤。 |
| `02_spatiotemporal_statistics/daily_safe_runner.py` | 运行日统计任务并处理检查点和重试。 |
| `02_spatiotemporal_statistics/daily_statistics_core.py` | 计算日优先像元和湖泊时间汇总。 |
| `02_spatiotemporal_statistics/daily_status.py` | 汇报日统计流程进度与状态。 |
| `02_spatiotemporal_statistics/daily_workflow.py` | 日统计流程的配置与运行入口。 |
| `02_spatiotemporal_statistics/pipeline.py` | 基于逐景或日值计算时间汇总。 |
| `02_spatiotemporal_statistics/requirements.txt` | 时间统计依赖包列表。 |
| `02_spatiotemporal_statistics/statistics_core.py` | 统计共用的汇总和质量控制函数。 |
| `02_spatiotemporal_statistics/trend_statistics/requirements.txt` | 趋势表生成依赖包列表。 |
| `02_spatiotemporal_statistics/trend_statistics/trend_statistics.py` | 计算Sen斜率和Mann–Kendall统计量并写出表格。 |
| `02_spatiotemporal_statistics/verify_collection_statistics.py` | 核验集合级时间统计输入与输出。 |
| `02_spatiotemporal_statistics/verify_daily_statistics.py` | 核验日优先统计及代表性源像元。 |
| `02_spatiotemporal_statistics/verify_statistics.py` | 按配置规则核验时间统计结果。 |
| `02_spatiotemporal_statistics/year_month_engine.py` | 基于日值计算年月湖泊级和像元级汇总。 |
| `02_spatiotemporal_statistics/year_month_runner.py` | 规划并运行指定湖泊的年月统计。 |
| `LICENSE` | MIT许可证文本。 |
| `MANIFEST.csv` | 文件清单、文件大小和SHA-256校验值。 |
| `README.md` | 英文仓库说明及完整文件附录。 |
| `README_zh.md` | 中文仓库说明及完整文件附录。 |
