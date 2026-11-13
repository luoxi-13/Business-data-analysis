# 基于 Olist 电商数据的用户价值与履约时效分析

《商业数据分析》期末大作业。数据是 Kaggle 公开的 Olist 巴西电商数据集（2016-09 至 2018-08，96,478 条已送达订单），用 RFM、帕累托/ABC、SARIMA 三个模型分别回答三个问题：谁值得投钱、卖什么最赚钱、下个月能卖多少。结论最后做成了一屏能点的 Pyecharts 大屏。

王京京　数据科学学院 大数据242　2024459985210　2026-10-29

**仓库地址：[github.com/luoxi-13/Business-data-analysis](https://github.com/luoxi-13/Business-data-analysis)**　纸质版大作业已提交，代码、30 张图和交互大屏都放在这个仓库里。

**在线交互大屏 → [点这里打开](https://luoxi-13.github.io/Business-data-analysis/)**

浏览器里直接跑，能切年份、点州下钻、拖时间窗，不用装环境、不用下数据。本机看就双击 `outputs/dashboard.html`（ECharts 已内联，断网也能开）。

> 链接第一次点开若是 404，按文末「在 GitHub 上打开交互大屏」开一次 Pages 就行，设置一次以后一直有效。

---

## 先看结论

**客户分层很清楚，但高价值的人太少。** 93,358 个客户里，复购频次的中位数是 1——绝大多数人只买过一次。消费金额高于中位数的那批客户撑起了大部分销售额：其中人数占 24.1% 的"重要发展客户"贡献 38.2%，"重要挽留客户"再贡献 37.1%。而平均消费 R$343.41（约为平台平均客单价的两倍）的重要价值客户只有 1,379 人，占 1.5%，销售额占 3.1%。不是没有高价值客户，是高价值客户太少、还几乎不回头。

**货也在少数品类里。** 18 个 A 类品类占品类数的 25%，贡献 81.3% 的销售额（约 1,254 万雷亚尔）；46 个 C 类品类占了 63.9% 的数量，加起来只有 9.1%。

**差评几乎都跟迟到有关。** 93.2% 的订单按时或提前送达，平均提前 11.87 天；剩下 6.8% 的延迟订单平均超期 10.6 天，评分只有 2.2 分，而按时送达的是 4.3 分。运费占比和评分的相关系数只有 −0.025，基本不相关——客户在意的不是运费贵，是付了钱没按时到。

一句话：这家平台不缺流量，缺的是把存量客户、品类结构和履约尾部管细的能力。下面是支撑这三句话的图。

---

## 数据

| 项目 | 说明 |
|---|---|
| 数据集 | Olist Brazilian E-Commerce Public Dataset |
| 来源 | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)，官方 CSV 下载（2026-10-24），未使用爬虫 |
| 许可证 | CC BY-NC-SA 4.0，仅用于教学与研究 |
| 规模 | 9 张表、99,441 条订单，清洗后保留已送达订单 96,478 条 × 41 列 |
| 派生字段 | 延迟天数、履约时长、运费占比 |

清洗以订单主表为核心做七表左连接，只保留 `order_status = delivered`；缺失评分填 0、类别填 unknown、产品属性取中位数；配送时长为负或超过 90 天的记录剔除。结果导出为 `data/clean/olist_master_clean.csv`，后面所有图和模型都从这一份主表算起。

> 原始 CSV 太大没直接传，仓库根目录放了打包好的 [data.zip](https://github.com/luoxi-13/Business-data-analysis/raw/master/data.zip)（约 65 MB，文件在 `master` 分支）。下载解压到项目根目录，按「[数据文件说明](#数据文件说明)」一节操作即可，清洗表跑代码时也会自动生成。

---

## 模型与结果

### 用户价值：RFM

R、F、M 三个维度按中位数打分，先画成 R×M 矩阵。颜色越深人数越多——四个象限几乎是均分的（24.4%–25.6%），这正是用中位数切分的结果：没有天然的"低价值大多数"，分层要靠后面的八分类继续往下切。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/models/M01_rfm_matrix.png" width="720">

*图 5-16　RFM 矩阵（R×M 视角）：左上角 R 好、M 好的象限人数最多，颜色越红人数占比越高*

八分类之后再并排看人数占比和销售额贡献，结构就出来了：1,379 名重要价值客户（1.5%）人均消费 R$343.41，是平台平均的两倍，但人数太少，销售额只占 3.1%；真正扛销量的是人数占 24.1% 的重要发展客户（38.2%）和 23.2% 的重要挽留客户（37.1%）——两群人加起来撑起四分之三的销售额。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/models/M02_rfm_structure.png" width="860">

*图 5-17　客群结构双图：左边是人数占比，右边是销售额占比，两边的排序对不上，就是运营的机会所在*

把全部客户按购买频次 × 消费金额铺成散点，各组的位置一目了然：紫色和绿色两大团挤在 F=1 的柱子上，金额拖出一条长尾；散落在右侧的那些孤点就是复购客户，最远的一个买了 15 次。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/fig04_rfm_scatter.png" width="860">

*图 4-1　RFM 用户分群散点图：F=1 的两大团是运营主战场，右侧孤点是复购样本*

### 商品结构：帕累托 / ABC

品类销售额从高到低累加，切出 A（0–80%）、B（80–90%）、C（90–100%）三段。柱子和累计曲线的距离越拉越大，就是长尾在变长。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/fig05_pareto_category.png" width="880">

*图 4-2　帕累托 / ABC 分类图：柱形是单个品类销售额，绿线是累计占比，两条虚线卡在 80% 和 90%*

从数量看更刺眼：18 个 A 类占品类总数的 25%，拿走 81.3% 的销售额；46 个 C 类占了 63.9% 的数量，加起来只剩 9.1%。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/models/M03_abc_composition.png" width="760">

*图 5-18　ABC 分类构成：左边品类数量占比，右边销售额占比，两边对着看差距就出来了*

分类结果直接翻译成动作，就是下面这张四象限图。A 类保现货、B 类按需补、C 类下架或按单采购——库存策略的差别比分析本身更值钱。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/models/M04_abc_strategy_matrix.png" width="720">

*图 5-19　ABC 库存策略矩阵：从"卖得好不好"到"该怎么备货"*

### 销售预测：SARIMA

先用季节分解把月销售额拆成趋势、季节、残差三层。季节项在 2017 年 11 月出现最大的正向偏离（黑五），3—4 月和 6—7 月是淡季；样本里只有一个完整年度周期，这一点直接决定了后面模型的误差水平。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/models/M05_sarima_decompose.png" width="860">

*图 5-20　序列分解：趋势项一路上行，季节项的峰值落在 2017 年 11 月*

网格搜索 AIC 之后选的是 SARIMA(0,1,0)×(0,1,1,12)。拿最后 3 个月（2018 年 5–7 月）做回测，MAPE 21.67%、RMSE 约 23.07 万雷亚尔，三个月误差在 +16% 到 +27.3% 之间——全是高估，不是随机抖动。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/models/M06_sarima_backtest.png" width="820">

*图 5-21　回测对比：蓝线是历史实际，红方块是模型预测，绿点是回测期的真实值，标题里标了每个月的误差*

预测 2018 年 8–10 月，9 月点预测 1,162,670 雷亚尔，95% 区间 [896,350, 1,428,990]。区间比点位更有用——下界用来锁采购资金，上界用来排临时人力。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/models/M07_sarima_forecast_ci.png" width="840">

*图 5-22　95% 置信区间预测：越往后带宽越宽，符合时序预测的规律*

把历史、回测、预测三段画在同一张图上，模型的脾气就看清了：蓝线是真实销售额，绿段是回测期的拟合，红段是未来三个月，两段阴影是各自的置信区间。回测那三个绿点全都压在蓝线上方 1.2–1.3 百万的位置，而实际只有 1.01–1.13 百万——高估是系统性的，用的时候得往下打折。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/fig06_sarima_forecast.png" width="880">

*图 4-3　月度销售额预测总图：历史 / 回测 / 未来三段拼接，标题标注 MAPE=21.67%、RMSE=230,650*

---

## 图表

论文一共 30 张图：第 4 章的 3 张（RFM 散点、帕累托、SARIMA 预测，上一节已经贴过），第 5 章的 27 张——探索性分析 3 张、图表模板 12 张、业务模型图 7 张、大屏截图 5 张。按顺序过一遍。

图都存放在 `outputs/figures/`，README 里用 CDN 绝对地址引用。之前写的是仓库相对路径，本机预览没问题，但换台电脑看 GitHub 网页时图片要从 `raw.githubusercontent.com` 加载，这个域名在不少网络里连不上，就会出现整页裂图；改成绝对地址后任何机器打开都能正常加载。离线看图直接翻 `outputs/figures/` 目录。

### 探索性分析

订单量两年翻了两个数量级。2017 年 11 月黑五冲到 7,289 单的峰值，2018 年起进入 6,100–7,100 单的平台期，不再大起大落。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/fig01_monthly_orders.png" width="860">

*图 5-1　每月订单量：2017-11 的尖峰就是黑五，之后是一段稳定的高位平台期*

品类份额比较平，头部五个加起来 39.2%，第一名健康美容 9.1%，没有哪个品类能单独撑起平台，这也是为什么后面要做品类分群而不是押爆款。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/fig02_category_sales_share.png" width="700">

*图 5-2　类别销售额占比：Top10 合计 62.4%，分布相当平均*

延迟分布的均值是 −11.87 天，也就是说平均提前 12 天送达。但右侧那条尾巴拖到 30 天以上，这就是客诉的来源。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/fig03_delivery_delay_dist.png" width="840">

*图 5-3　延迟天数分布：主体在 −20 至 0 天，长尾才是要管的部分*

### 12 种图表模板

对照《Excel 数据可视化》教材复现的，从折线到双轴组合一共 12 种，超过作业要求的 10 种。每个模板都不是随手画，而是对应一种数据关系：时间序列用折线面积、类别排序用条形、构成用饼和环、相关用散点、分布用直方和箱线、多维比较用气泡和热力。

| | | |
|---|---|---|
| <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T01_line_monthly_orders.png" width="260"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T02_area_cumulative_sales.png" width="260"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T03_column_state_orders.png" width="260"> |
| T01 折线　每月订单量 | T02 面积　月销售额 | T03 柱形　各州订单量 |
| <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T04_bar_top10_category.png" width="260"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T05_pie_payment_type.png" width="260"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T06_donut_review_score.png" width="260"> |
| T04 条形　类别销售额 | T05 饼图　支付方式 | T06 环形　评分分布 |
| <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T07_scatter_freight_review.png" width="260"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T08_bubble_state.png" width="260"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T09_hist_delivery_duration.png" width="260"> |
| T07 散点　运费占比 vs 评分 | T08 气泡　州订单-客单价-销售额 | T09 直方　配送时长 |
| <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T10_box_score_duration.png" width="260"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T11_heatmap_corr.png" width="260"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T12_combo_sales_review.png" width="260"> |
| T10 箱线　各评分配送时长 | T11 热力　指标相关矩阵 | T12 双轴　销售额 × 评分 |

这几张里信息量最大的是 T10 和 T11。箱线图按评分分组看配送时长，5 星中位数 9 天、1 星 16 天，上四分位 29 天；热力图里和评分负相关最强的也是配送时长（−0.332），其次是延迟天数（−0.267）。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T10_box_score_duration.png" width="700">

*图 5-13　箱线图：评分越低，箱子越长、位置越高*

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/template/T11_heatmap_corr.png" width="620">

*图 5-14　相关矩阵：金额、运费、件数之间强相关，与评分几乎无关*

### 交互大屏

`outputs/dashboard.html` 是用 Pyecharts 搭的，10 个页签。ECharts 已经内联进 HTML（2 MB），不联网、不起服务。打开方式有两种：本机双击这个文件，或者直接点 README 顶部的[在线链接](https://luoxi-13.github.io/Business-data-analysis/)——打开的是同一份文件，只是放到了 `docs/index.html` 供网页访问。

第一个页签是使用说明，把三步上手和三级联动写在明面上，省得别人打开大屏先愣住。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/dashboard_screenshot/dash_guide.png" width="800">

*使用说明页：三步上手 + 三级联动规则（论文里没占图号）*

首页做成了四图同屏的驾驶舱，一屏回答四个问题：卖得怎么样、卖到哪里、怎么付的钱、客户满不满意。

<img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/dashboard_screenshot/dash_cockpit.png" width="780">

*图 5-23　驾驶舱总览：月度趋势 / 各州 Top5 / 支付占比 / 评分分布*

点进单个页签，交互分三层：年份时间轴切换（2016、2017、2018、全部，数据在 Python 端预先聚合好，切的时候没有延迟）；点州柱条触发 click 事件，把选中的州写进全局状态，给其他图做下钻；再往下就是 Tooltip、DataZoom 缩放和散点的 VisualMap 配色。

| | |
|---|---|
| <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/dashboard_screenshot/dash_overview.png" width="400"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/dashboard_screenshot/dash_state.png" width="400"> |
| KPI 指标卡：订单 / 销售额 / 准时率 / 评分 | 各州销售额 Top10，内嵌年份 Timeline |
| <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/dashboard_screenshot/dash_month.png" width="400"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/dashboard_screenshot/dash_scatter.png" width="400"> |
| 月度趋势，底部 DataZoom 可拖时间窗 | 延迟 × 评分散点，VisualMap 红黄绿映射 |
| <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/dashboard_screenshot/dash_category.png" width="400"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/dashboard_screenshot/dash_payment.png" width="400"> |
| 类别销售额 Top10，健康美容 1.41 百万居首 | 支付方式，信用卡 76.64%、boleto 19.89% |
| <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/dashboard_screenshot/dash_duration.png" width="400"> | <img src="https://cdn.jsdelivr.net/gh/luoxi-13/Business-data-analysis@main/outputs/figures/dashboard_screenshot/dash_score.png" width="400"> |
| 配送时长分布，6–10 天最集中（32,855 单） | 评分分布，5 星接近六成 |

散点那张值得多看两眼：评分 5 分的绿点在延迟 0 天左侧连成一条实线，越过 0 天明显变稀，而 1–2 分的红橙点一直铺到 +25 天——相关性是 −0.267，方向很清楚，只是没有想象的那么"断崖"。这一页底部同样带年份 Timeline。

页签清单：

| 页签 | 内容 |
|---|---|
| 使用说明 | 三步上手指引 + 联动说明 |
| 驾驶舱总览 | Grid 四图同屏 |
| KPI 总览 | 总订单 96,478 / 总销售额 R$15.4M / 准时率 93.2% / 平均评分 4.16 |
| 各州销售额 Top10 | 横向柱状，可点击下钻 |
| 类别销售额 Top10 | 横向条形 |
| 支付方式占比 | 饼图，信用卡 76.6%、boleto 19.9% |
| 月度销售额趋势 | 面积折线 + DataZoom |
| 客户评分分布 | 环形图，5 星 58.2%、1 星 9.8% |
| 配送时长分布 | 直方图，中位数 10 天 |
| 延迟 × 评分散点 | VisualMap 红黄绿 |

**为什么在 GitHub 仓库里点这个 HTML 打不开？** GitHub 出于安全考虑，把仓库里的 `.html` 当源码展示，不会执行里面的 JavaScript；而 `file:///C:/Users/...` 只是你自己电脑上的路径，换台机器就不存在了。想让别人点一下就能进，只能把这份 HTML 当网页发布出去——GitHub Pages 就是干这个的，免费、不用服务器。设置方法在「在 GitHub 上打开交互大屏」一节。

---

## 目录

```
Olist_Business_Analysis/                # 项目根目录
├── Olist_Business_Analysis.ipynb       # 核心分析笔记本（含代码、图表与业务解读）
├── run_all.py                          # 一键复现入口脚本（转调 scripts/run_all.py）
├── requirements.txt                    # Python 依赖库列表
├── .gitignore                          # 排除 data 下的大 CSV 与临时文件
├── README.md                           # 项目简介、数据来源、复现步骤与图表索引
├── 《商业数据分析》期末大作业.docx       # 最终报告文档
│
├── data/                               # 数据文件夹
│   ├── raw/                            # 9 张原始 CSV 表（未入库，见下节）
│   └── clean/                          # 清洗后的主表 olist_master_clean.csv（未入库）
│
├── outputs/                            # 输出结果文件夹
│   ├── dashboard.html                  # 交互大屏，ECharts 已内联，双击即可打开
│   ├── figures/                        # 自动生成的分析图表
│   │   ├── fig01–fig06                 # 第 4 章核心图（趋势/占比/延迟/RFM/帕累托/预测）
│   │   ├── models/M01–M07              # 业务模型图（RFM 2 张、ABC 2 张、SARIMA 3 张）
│   │   ├── template/T01–T12            # 12 种可视化模板复现
│   │   └── dashboard_screenshot/       # 大屏 10 个页签截图
│   └── tables/                         # RFM / ABC 结果表（CSV）
│       ├── rfm_summary.csv             # RFM 八分类汇总（客户数、平均 R/F/M）
│       └── abc_classification.csv      # ABC 分类明细（品类、销售额、累计占比）
│
├── scripts/                            # 辅助脚本与静态资源
│   ├── run_all.py                      # 一键复现入口（由 Notebook 导出）
│   └── echarts.min.js                  # 大屏所需的 ECharts 库文件
│
├── docs/                               # GitHub Pages 站点目录（在线大屏）
│   ├── index.html                      # 大屏在线版，与 outputs/dashboard.html 完全相同
│   ├── .nojekyll                       # 让 Pages 跳过 Jekyll、原样发布
│   └── .gitkeep                        # 目录占位
```

## 数据文件说明

原始 CSV 体积太大，GitHub 不收单文件 100 MB 以上的东西，全传上去 clone 也慢，所以 `.gitignore` 排掉了 `data/raw/*.csv` 和 `data/clean/*.csv`，改用打包文件：

👉 **下载数据集**：[data.zip（约 65 MB）](https://github.com/luoxi-13/Business-data-analysis/raw/master/data.zip)，解压到项目根目录即可，里面是 `data/raw/` 和 `data/clean/` 两层。

| 目录 | 内容 | 体积 |
|---|---|---|
| `data/raw/` | 9 张原始表，最大的是 `olist_geolocation_dataset.csv`（61 MB） | 121 MB |
| `data/clean/` | 清洗后的主表 `olist_master_clean.csv` | 51 MB |

其实只保留 `data/raw/` 就够了：打开笔记本从第 2 步跑起，或直接 `python run_all.py`，清洗表会自动重新生成。也可以跳过压缩包，去 [Kaggle 数据集页](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) 下载原始 CSV 放进 `data/raw/`，文件名保持原样；配好 Kaggle token 的话用命令行更快：

```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
```

`outputs/` 里的东西都是跑出来的产物，仓库里留着是为了不跑代码也能看图：`dashboard.html`、30 张分析图、RFM/ABC 结果表都在，重新运行会原地覆盖。

## 怎么跑

装依赖：

```bash
pip install -r requirements.txt
```

再打开 `Olist_Business_Analysis.ipynb`，`Restart & Run All`，从清洗一路跑到大屏，大约十几分钟（SARIMA 网格搜索最慢）。不想开 Notebook 的话执行 `python run_all.py`（或 `python scripts/run_all.py`）也行，内容一样。

注意：**`data/raw/` 里没有 CSV 的话第一步就会报错**，先按上一节把数据放进去。

看大屏：点 README 顶部的在线链接，或双击 `outputs/dashboard.html`。要重新生成就在笔记本最后一个 cell 跑完，文件会覆盖到同一个位置；在线版还得同步一次到 `docs/index.html`，方法见下一节。

依赖：pandas、numpy、matplotlib、seaborn、statsmodels、pyecharts，Python 3.10 以上。中文字体依赖系统的 SimHei 或 Microsoft YaHei，Linux 上要单独装字体，否则图上全是方块。

## 在 GitHub 上打开交互大屏（设置一次，以后一直有效）

在线地址是 **https://luoxi-13.github.io/Business-data-analysis/**，这是 GitHub Pages 给本仓库分配的站点。大屏文件已经放进 `docs/index.html`，只差把 Pages 打开：

1. 打开仓库 → **Settings** → 左侧 **Pages**；
2. **Build and deployment** 的 Source 选 **Deploy from a branch**；
3. Branch 选 **`main`**（仓库默认分支就是 main），目录选 **`/docs`**，点 **Save**；
4. 等 1–2 分钟，再点 README 顶部的链接就能进。之后每次 push 会自动重新发布。

仓库需要是 **Public**，私有仓库的 Pages 要付费版。

把这些文件推上去。仓库默认分支是 `main`，本地如果还停在 `master`，先切过去（`git checkout main`，本地没有就 `git checkout -b main origin/main`）——推错分支的话 GitHub 首页显示的还是旧 README，等于没改：

```bash
git add .
git commit -m "加入在线交互大屏"
git push origin main
```

如果你之前在 GitHub 网页上改过 README，push 被拒绝时先 `git pull --rebase origin main`，冲突处保留本地这份（已经把线上新增的 data.zip 说明合进来了），再推一次。

**大屏改版之后**，笔记本只会更新 `outputs/dashboard.html`，在线版要手动同步：

```bash
cp outputs/dashboard.html docs/index.html
git add docs/index.html && git commit -m "同步大屏" && git push origin main
```

Pages 还没开、又想马上给别人看，可以先用这个临时预览地址（本仓库默认分支是 `main`）：
https://htmlpreview.github.io/?https://github.com/luoxi-13/Business-data-analysis/blob/main/outputs/dashboard.html
它加载 2 MB 的页面偶尔会不全，正式场合还是用 Pages。

## 两点说明

论文里 SARIMA 只用 19 个月训练，不到两个完整年度，季节因子估得不够准，MAPE 21.67% 偏高。这一点写在论文的局限性里了——引入 XGBoost 或者把促销日历当外生变量，应该还能往下压。

代码框架用了 AI 编程助手（Claude Code）辅助，主要是 Pyecharts 的 Timeline 联动和多图 Grid 布局那部分；模型选型、阈值设定和商业结论是自己做的，论文 7.3 节有披露。
