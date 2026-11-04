# 📊 Olist 巴西电商商业数据分析项目

> 基于 Kaggle Olist Brazilian E-Commerce Dataset 的商业数据分析大作业：数据清洗、RFM 用户价值分析、帕累托 ABC 分类、SARIMA 时间序列预测、12 种可视化模板复现、7 张业务模型图与 Pyecharts 交互式大屏（全 Python 实现）。

**作者**：王京京 ｜ **学号**：2024459985210 ｜ **学院**：数据科学学院 大数据242 ｜ **完成日期**：2026-10-29

---

## 📋 数据来源合规声明

| 项目 | 说明 |
|------|------|
| 数据集名称 | Olist Brazilian E-Commerce Public Dataset |
| 数据来源 | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| 获取方式 | Kaggle 官方公开数据文件（CSV），**非爬虫获取** |
| 下载日期 | 2026年10月24日 |
| 许可证 | CC BY-NC-SA 4.0（仅供学术研究使用） |
| 合规声明 | 本项目严格遵守数据使用协议，仅用于教学与研究目的，未使用任何网络爬虫技术（Scrapy/Selenium/requests 等） |

---

## 🗂 项目结构

```
Olist_Project/
├── Olist_Business_Analysis.ipynb   # ★ 主笔记本（全流程，已完整执行，含全部输出）
├── data/
│   ├── raw/                          # 原始数据（9个CSV，Kaggle下载）
│   │   ├── olist_customers_dataset.csv
│   │   ├── olist_geolocation_dataset.csv
│   │   ├── olist_order_items_dataset.csv
│   │   ├── olist_order_payments_dataset.csv
│   │   ├── olist_order_reviews_dataset.csv
│   │   ├── olist_orders_dataset.csv
│   │   ├── olist_products_dataset.csv
│   │   ├── olist_sellers_dataset.csv
│   │   └── product_category_name_translation.csv
│   └── clean/
│       └── olist_master_clean.csv    # 清洗后主表（96,478行 × 41列，50.25MB）
├── notebooks/                        # 分步笔记本（01清洗 / 02EDA / 03建模）
├── scripts/
│   ├── 01_load_and_clean.py          # Step1: 数据加载、七表连接、清洗
│   ├── 02_eda_visuals.py             # Step2: EDA 可视化（fig01–fig03）
│   ├── 03_business_models.py         # Step3: RFM/ABC/SARIMA 三大模型（fig04–fig06）
│   ├── 04_dashboard.py               # Step4: Streamlit 大屏（备选方案）
│   └── run_all.py                    # 一键复现脚本
├── outputs/
│   ├── dashboard.html                # ★ Pyecharts 交互大屏（928KB，浏览器直接打开）
│   ├── figures/
│   │   ├── fig01–fig06               # EDA + 模型核心图（6张，≥150dpi）
│   │   ├── template/                 # ★ 5.2 模板复现图（T01–T12，12种图表类型）
│   │   ├── models/                   # ★ 5.3 业务模型图（M01–M07，对应第4章）
│   │   └── dash_*.png                # ★ 5.4 大屏 8 个 Tab 页截图
│   └── tables/
│       ├── rfm_summary.csv           # RFM 八分类汇总表
│       └── abc_classification.csv    # ABC 分类结果表
├── docs/
│   └── 期末论文核心内容.md
├── 期末论文_基于Olist电商数据的用户价值与履约时效分析.docx
├── 《商业数据分析》期末大作业_成品.docx   # ★ 最终提交版（含31张图表）
├── requirements.txt
└── README.md                         # 本文件
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行主笔记本（推荐）

用 Jupyter Notebook / JupyterLab 打开 `Olist_Business_Analysis.ipynb`，
`Kernel → Restart & Run All` 一键复现全部结果（数据清洗 → EDA → 三大模型 → 12 种模板复现 → 7 张模型图 → Pyecharts 大屏）。也可直接查看已执行的输出。

### 3. 脚本方式一键复现

```bash
python scripts/run_all.py
```

依次执行：数据清洗 → EDA 图表 → 三大模型与结果表。

### 4. 查看交互式大屏

```bash
# 方式一：直接用浏览器打开已生成的大屏文件
outputs/dashboard.html

# 方式二：重新生成（在主笔记本最后一个 cell 运行后自动输出）
# 方式三：Streamlit 备选大屏
streamlit run scripts/04_dashboard.py
```

---

## 📈 核心成果

### 商业分析模型（第4章）

| 模型 | 关键参数 | 核心结论 | 对应模型图 |
|------|----------|----------|------------|
| RFM 用户价值分析 | 中位数阈值八分类 | 1,379 名"重要价值客户"（1.5%）贡献约 12% 销售额，平均消费 R$343.41 | M01、M02 |
| 帕累托/ABC 分类 | 80/20 原则切分 | 18 个 A 类（25%）贡献 81.3% 销售额；46 个 C 类（63.9%）仅贡献 9.1% | M03、M04 |
| SARIMA 预测 | (0,1,0)×(0,1,1,12)，AIC 网格搜索 | 回测 MAPE=21.67%，RMSE=R$230,650；预测 2018 年 8–10 月月均约 R$1.16M | M05–M07 |

### 可视化成果（第5章）

| 板块 | 数量 | 说明 | 输出位置 |
|------|------|------|----------|
| EDA 分析图 | 3 张 | 订单趋势 / 类别占比 / 延迟分布 | `fig01–fig03` |
| 5.2 模板复现 | **12 种** | 对照《Excel数据可视化》：折线、面积、柱形、条形、饼图、环形、散点、气泡、直方、箱线、热力、双轴组合 | `template/T01–T12` |
| 5.3 业务模型图 | **7 张** | 与第4章 RFM/ABC/SARIMA 一一对应 | `models/M01–M07` |
| 5.4 交互大屏 | **9 组件** | Pyecharts 全 Python 实现，三级联动 | `dashboard.html` + 8 张截图 |

### 交互大屏结构（10 个页签，全 Python 实现）

| 页签 | 内容 |
|------|------|
| 使用说明 | 三步上手指引 + 三级联动说明 |
| **驾驶舱总览** | Grid 四图同屏：月度销售额趋势 / 各州销售额 Top5 / 支付方式占比 / 客户评分分布 |
| KPI 指标卡 | 总订单数 / 总销售额 / 准时率 / 平均评分 |
| 各州销售额 Top10 | 横向柱状图，支持点击联动 |
| 类别销售额 Top10 | 横向条形图 |
| 支付方式占比 | 饼图 |
| 月度销售额趋势 | 面积折线图 + DataZoom 缩放 |
| 客户评分分布 | 环形图 |
| 配送时长分布 | 直方图 |
| 延迟×评分散点 | 散点图，VisualMap 红黄绿映射 |

> `dashboard.html` 约 2MB，ECharts 已内联，**双击即可离线打开**。

### 交互大屏联动机制（三级）

1. **① 年份 Timeline 时间轴联动**：7 个图表组件均内嵌"2016/2017/2018/全部"时间轴，点击即切换该年份重新聚合的数据（Python 层预聚合 4 套数据集，零延迟）；
2. **② 州柱状图 JS click 二级联动**：点击州柱条触发 ECharts click 事件，写入 `window.__selectedState` 全局状态（事件总线），支持其他图表订阅下钻；
3. **③ 悬浮与缩放交互**：全组件 Tooltip 悬浮详情、月趋势页 DataZoom 时间窗缩放、散点页 VisualMap 按评分红黄绿映射。

> 未使用 Power BI / Tableau，全流程 Python 实现。

---

## 🔑 关键发现

- **93.2%** 订单准时或提前送达（平均提前 11.87 天），但 **6.8%** 延迟订单平均延迟超 15 天，评分仅 2.8 分（整体 4.1 分）——物流时效是满意度首要因素；
- 平台复购率极低（F 中位数 = 1），1.5% 高价值客户是核心利润源，LTV 挖掘空间大；
- 25% 的 A 类产品是平台"现金牛"，63.9% 的 C 类长尾 SKU 仅贡献 9.1% 销售额，存在明显优化空间。

---

## 🔧 技术栈

- **数据处理**：Pandas, NumPy
- **可视化**：Matplotlib, Seaborn, Plotly, **Pyecharts 2.1.0**（大屏）
- **统计建模**：Statsmodels（SARIMAX / seasonal_decompose）
- **交互**：ipywidgets（笔记本内联动）、ECharts Timeline/事件总线（大屏联动）
- **Python 版本**：3.10+

---

## ⚠️ 注意事项

1. 原始数据（9 个 CSV）需放置于 `data/raw/` 目录下；
2. 图表中文显示依赖系统字体（SimHei / Microsoft YaHei），Linux 需额外配置字体；
3. SARIMA 训练数据仅 19 个月，季节效应估计不充分，MAPE 约 21.67%（详见论文 7.2 局限性）；
4. `outputs/dashboard.html` 为自包含 HTML（ECharts JS 已内联），离线可直接打开；
5. AI 使用披露：AI 编程助手仅辅助代码框架生成，核心业务逻辑、模型选型与商业结论由作者独立完成（详见论文第 7.3 节）。
