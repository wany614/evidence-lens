# Evidence Lens 研究与发布验收

本轮交付三个独立版本。A 提供有限图像尺寸的通用 CPU 工具。B 用真实训练的新小头和两个 ETH3D 确认场景，分别检验有限查询能找到多少来源、模型能否排出更少错误的像素。C 提供七个原创目标画面、独立评价和三份理论阅读卡。每个版本均有独立的提交、原始运行清单和公开附件。

## A · 基础封存

公开标签 `v0.5.0` 指向 `c259c987b77296f368d78a9b09ef504226ac6cc3`。A_BASELINE_ID 为 `sha256:6e8e837afa0818cdf8c100b494672b46c10bdbd22e522f388983543f0a5b0f23`。CPU wheel 为 405,919 字节，SHA-256 为 `40d9b6ffb9d881c63d4710fee71112920dd665da831ef744fb062a20f86de734`。八件附件匿名回读共 137,865,941 字节。模型参数沿用 v0.3.0-rc.1，包括真实微调 MS-ft。 [A 发布及下载](https://github.com/wany614/evidence-lens/releases/tag/v0.5.0) · [分发回执](../verification/v0.5.0/distribution.json) · [既有四模型参数](https://github.com/wany614/evidence-lens/releases/download/v0.3.0-rc.1/evidence-lens-cpu-v0.3.0-rc.1.zip)

新分析网格为 256×256、448×252、252×448、378×252 和 375×251，旧版 256 分析／252 编码路径仍可重放。打包源码的 59 项测试通过。Mac arm64 与 Windows x86-64 上，同机旧版与新版对四法的历史输出逐位一致。跨平台两种线性方法每法 42/42 视角逐位相同；MS-ft 有两个像素跨旧阈值。Linux arm64 离线验收覆盖基础 CPU 安装和 42 个视角。旧 T1 仍缺 48/288 项，C6 两套阈值均为空。含 16,251 个高斯的真实资产通过原生 CUDA 渲染复核。 [验收与已知差异](../releases/v0.5.0/ACCEPTANCE.md) · [平台矩阵](../verification/v0.5.0/README.md)

## B · 多来源 CPU 质检

公开实验标签 `many-source-v0.1.0-dev.0` 指向 `19cd7079ee5d4e470af4ca6b2543ad916ef0a4b4`。科学代码在开封前固定为 `6e55dd6969b58faa08dc4a7033d7c6687a919dad`。15 次 gsplat 拟合全部成功，均使用同一上游提交。每次取样 6,000 次，保存的 checkpoint 含 5,999 次参数更新。1,053 项来源自渲染和 102 项目标预测与原生实现逐值核对。六个确认 checkpoint 的 44 项目标输出封存后，独立评价器才首次解码 32 张不同的确认参考图。 [上游结果](../research/many-source-v1/upstream.json) · [逐来源更新](../research/many-source-v1/checkpoint-updates.json) · [开封与参考访问](../research/many-source-v1/access-order.json)

| 数据角色 | 固定家族与来源 |
|---|---|
| 原创系数拟合 | staggered arcade、terraced court、woven truss。每家族分别以 16、128 张来源重建，共六个 checkpoint、36 个目标单元。 |
| 开发选择 | 原创 layered louvers 及 ETH3D delivery_area，后者 R=A=34、H=10。开发选定 L2=0.1 与 residual 简单规则。 |
| 确认 | 原创 perforated gallery、corner alcoves 各用 R16/R128、每重建六个目标。ETH3D living_room 与 bridge 分别 R=A=55、100，各留出 10 张 H。 |

真实 NumPy 风险头使用 144,000 行训练数据，其中 6,833 行属于已登记 RGB 错误。三个正则候选、一次最终拟合，产出 44 个系数和一个截距。含预处理的模型文件为 6,382 字节，SHA-256 为 `b9e6b761f2ff15d2c2aada851aa0e239f550163368107692b0c536843f11c721`。新分数用于排序，阈值为空。B 的校准拟合与 DINOv2 更新均为零。 [权重与模型卡](../research/many-source-v1/model-card.md) · [安装及 CPU 命令](../research/many-source-v1/README.md) · [真实审计示例](../../examples/many-source/b04-layered-louvers/audit-bank/report.html)

| 已开封的外部场景 | 超阈错误／有标签像素 | P1 固定四源 → 覆盖四源漏掉的正确像素／全部正确像素 | P2 新头／残差接纳的错误数 | 两法各自的接纳量／共同域 | 新头错误率变化 |
|---|---:|---:|---:|---:|---:|
| living_room | 332,811 / 952,560 | 141,729 → 153 / 619,749 | 51,334 / 45,524 | 190,510 / 952,560 | +3.050 个百分点 |
| bridge | 341,069 / 952,560 | 139,207 → 21 / 611,491 | 44,943 / 46,014 | 190,510 / 952,560 | −0.562 个百分点 |

P1 只数独立颜色预测正确、所查源均不可比、完整来源库另有可比源的像素。覆盖优先的 K16 分别还漏掉 1/619,749 和 0/611,491。在相同预测、K16、查询集合和共同可评分域下，各目标接纳最低风险的 20%（向下取整）。living_room 的新头／残差 AP 为 0.4454／0.4518，同域基率 0.3494；bridge 为 0.4533／0.3646，基率 0.3581。十个视角在场景内等权。P2 的预登记结论为 **completed_negative**。 [全实验结果](../research/many-source-v1/results.md) · [逐视角分母](../research/many-source-v1/views.csv) · [交互结果图板](../../examples/many-source-study/report.html)

相同覆盖 Q4 下，living_room 的新头／MS-L1／MS-ft 共域 AP 分别为 0.4503／0.4428／0.4223（330,962/950,371 个正例）；bridge 为 0.4476／0.4519／0.4490（341,057/952,527）。同一新头在其自身域上按 K4/8/16/32/全库排序，living_room 的 AP 为 0.4522/0.4517/0.4454/0.4384/0.4372，bridge 为 0.4476/0.4535/0.4533/0.4537/0.4480。两场景均有 952,560 个有限新头分数。这里有两个外部确认场景，每个场景内的相邻视角和像素均有关联。 [三种比较域和基率](../research/many-source-v1/scene-domain-averages.csv) · [22,584 行原因及计数](../../examples/many-source-study/unknown-reason-counts.csv) · [已生成视角的检查代理任务](../research/many-source-v1/view-priority.csv)

Mac 对原创 R16/K16 的新头，首个结果耗时 0.376–0.382 秒，其中首次来源准备 0.0810–0.0813 秒；七次热运行的中位数在 0.1665–0.1674 秒之间。另一次 R128 残差全库扫描耗时 1.841 秒，读取 218,443,586 字节来源数据，进程 RSS 峰值 347,308,032 字节。Mac 与离线 Linux 对公开案例两种模式的 95,256 个有限分数、原因、证据来源和配额成员逐值一致。重建使用 Windows CUDA。外部相机为 ETH3D 提供的已知标定，使用各场景的标定单位；没有独立验证其米制换算。 [热／冷成本](../research/many-source-v1/deployment/README.md) · [上游与适用范围](../research/many-source-v1/limitations.md)

## C · 人文模型与理论基础

`humanities-foundation-v1` 引用上述两个提交。读者可以先看铃架六张目标图，再看四种旧法的风险和实际查询来源。固定廊道目标使用同一原生 RGB、alpha、Z 文件，五档查询的文件 SHA-256 均为 `c5eb81ece005d7ab313aa2151815deba16f5d71306867a120e9d9af34d57f2f0`。图库、风险、状态与独立评价来自原创合成相机。 [打开离线案例](../../examples/evidence-foundation/index.html) · [案例索引](../../examples/evidence-foundation/case-index.json) · [C 版本及离线下载](https://github.com/wany614/evidence-lens/releases/tag/humanities-foundation-v1)

廊道共有 4,579/95,256 个独立 RGB 错误。对其余 90,677 个正确像素，覆盖优先查询 4/8/16/32/128 个来源分别漏掉 180/120/109/12/0 个可比证据。固定四源在该视角漏 74 个，少于覆盖四源的 180 个。完整库仍有 220/95,256 个位置未找到可比证据。全库相对自身的遗漏为零是定义结果。独立程序复算 132 个源视图、240 张离线 PNG 和 29 个条件的 87 张排名／状态／原因图；浏览器另行检查了 7 组独立评价。 [数据复核](../humanities-foundation/case-data-check.json) · [离线 Chrome 验收](../../examples/evidence-foundation/viewer-check.json)

理论卡分别核验 Bolter 与 Grusin 1996 年期刊 pp. 313–314、London Charter 英文 2.1 第 4.4–4.7 节及两个术语、Seville 2017 英文第 4 和 7 原则。媒介再中介、图像误差、查询所得证据、过程可查性与历史真实性各有对应的材料和边界。Wany 保留论文选题与论证选择。外部评审提出四图可能漏证据，ChatGPT 将其变成实验条件，Claude 复核所给材料和代码，Codex 执行与核验。 [理论卡与原文](../humanities-foundation/THEORY_SOURCES.md) · [贡献记录](../humanities-foundation/CONTRIBUTIONS.md) · [Research Log](https://docs.google.com/document/d/1-aLBWAepKhZd6Wpv_CW1x8Hfds5-9DAePKEfhpftKak/edit)

## 归档、权利与停止点

B 科学与部署结账口径共 188 个终态，其中六项失败。公开草稿首次建立时列表尚未同步，保留失败记录并在同一草稿上重试成功。另有一单零拟合的 Windows 环境清点遇到接收端父件归属问题，随后完成独立核对。上游训练及确认对象的失败均为零。外部下载共 2,419,531,415 字节。科学结账中的 15 次 GPU 工单按完整记录器墙钟计 1,090.73 秒；CPU 计 1,978.42 核秒，未采样区间另预留不超过 3,874.22 核秒。Mac 侧 B 终态归档实际占用 12,485,156,864 字节，Windows 六个确认工单新增项目分配量为 2,165,888,008 字节。不同位置的归档、传输副本和工作目录分别计量。 [逐运行账本](../research/many-source-v1/resources-and-failures.md) · [Windows 确认磁盘](../research/many-source-v1/windows-confirmation-storage.json)

原创代码与新头参数按 Apache-2.0，原创展示和可授权的实测资料按 CC BY 4.0；旧模型组件沿其既有许可。ETH3D 使用 CC BY-NC-SA 4.0，公开仓库提供来源、数值与哈希。第三方照片及派生图、理论原文的副本保存在研究归档。专精模型、真实控制、观众研究、历史真实性验证和论文定题均留待后续。 [分项权利说明](../research/many-source-v1/rights.md) · [七层判断与反例](../humanities-foundation/CONCEPTS_AND_CLAIMS.md)

### 核心运行索引

| 阶段 | 具体 run ID | 对照凭证 |
|---|---|---|
| A 基线封存／匿名回读 | `20261124T111058386191Z-d99a4c418276454c` ／ `20261124T111130398649Z-9f07d2bcce9b4ad7` | [A 文件清单](../releases/v0.5.0/BASELINE_MANIFEST.json) |
| B 最终拟合／确认前冻结 | `20261124T115857282953Z-3f6e989f5d4844b1` ／ `20261124T120332809100Z-ec53d7a4a3b94042` | [冻结文件](../research/many-source-v1/freeze-manifest.json) |
| B 统一开封／评价结果 | `20261124T122014191481Z-ddac5a608d5a4d90` ／ `20261124T122203772897Z-ba03676b98d64b31` | [开封](../research/many-source-v1/opening.json) · [结果](../research/many-source-v1/endpoints.json) |
| B 发布封存／匿名回读 | `20261124T122658772178Z-8af722eec6fa438e` ／ `20261124T122758951718Z-0096333565594edf` | [B 文件清单](../research/many-source-v1/B_MANIFEST.json) · [九个附件，共 91,436,242 字节](../research/many-source-v1/anonymous-readback.json) |
| C 三份原文／原创案例提取 | `20261124T123002024983Z-c2f5815f390d4061` ／ `20261124T123629027571Z-35e7d606002541df` | [所读版本](../humanities-foundation/primary-editions.json) · [案例文件](../../examples/evidence-foundation/case-files.json) |
| C 独立数据复核／浏览器验收 | `20261124T123844169590Z-9e19431a2a234675` ／ `20261124T124407786388Z-6f6d4b8266a442c9` | [数值回执](../humanities-foundation/case-data-check.json) · [界面回执](../../examples/evidence-foundation/case-viewer.json) |
| C 封存／匿名回读 | `20261124T124551683640Z-92178433fcf54568` ／ `20261124T124626017029Z-cb14be74fd9743f9` | [C 文件清单](../humanities-foundation/C_MANIFEST.json) · [四个附件，共 122,362,821 字节](../humanities-foundation/anonymous-readback.json) |
| C 匿名下载件的断网重放 | `20261124T124747170789Z-1322fd81aebc4dc6` | [真实 ZIP 解压后再次通过 29 个条件与七组评价](../humanities-foundation/public-offline-replay.json) |
| 全部 15 个 Windows 工单的 D 盘实物结算 | `20261124T125054498049Z-2b2cf8a105114b8c` | [期末空间、归属范围及逐项身份](windows-final-storage.json) |

C 标签的提交为 `23d05371d3c8f2a09afb3a6dcf633513b0e457e4`。可单独下载 8,785,767 字节的离线展页，或再下载 113,570,300 字节的原创来源与浮点资料复核原数值。两份 ZIP 均经 GitHub 标签与匿名 SHA-256 回读验证。 [离线查看器](https://github.com/wany614/evidence-lens/releases/download/humanities-foundation-v1/evidence-lens-foundation-viewer-v1.zip) · [原创复核数据](https://github.com/wany614/evidence-lens/releases/download/humanities-foundation-v1/evidence-lens-foundation-source-photos-v1.zip)

以 C 匿名回读完成为截止，Mac 接收端可核对 267 个不同终态原生 run。A 为 48 成功、9 失败，B 为 192 成功、7 失败，C 为 11 成功、0 失败。这个数字包含验收和发布操作，表中另列未进入同一终态目录的 Windows 环境清点接收记录。包含发布收尾时，Mac 的完整 B 原生归档实际占用 12,610,924,544 字节，15 次 GPU 工单累计按完整墙钟结账 1,090.73 秒。CPU 核算 1,980.99 核秒，全部未采样区间按逻辑核饱和计的附加上界为 4,114.66 核秒。归档统计和匿名 ZIP 再重放均发生在截止时刻之后。 [全 267 条运行及失败](run-ledger.json)

原 Research Log 已逐轮追加 RL-025 至 RL-032，并从 Google 文档读回，八个条目各出现一次。B 阶段的一次 Claude Opus max 调用超时；之后实际取得 Opus high 的代码复核。在 C 的两次 Opus high 复核中，Claude 依据收到的文稿和代码指出合成来源的表述、哈希所能证明的范围及打包前后父件一致性问题。Codex 对原文、代码与原始数组进行检查并修正。 [Research Log](https://docs.google.com/document/d/1-aLBWAepKhZd6Wpv_CW1x8Hfds5-9DAePKEfhpftKak/edit) · [贡献与审阅的职责范围](../humanities-foundation/CONTRIBUTIONS.md)

Windows D 盘的期末清点用卷与文件 ID 跨目录去重。15 项拟合、其本地回传 ZIP 与队列控制占 5,153,342,648 字节，单列的依赖清点占 532,096 字节，合计 5,153,874,744 字节、5,050 个唯一文件。此前六单的增量已包含在这个存量中。缺少全 B 开始时的完整物理快照，共享运行环境的净变化也无法分摊。15 个云端回件为 2,392,041,507 逻辑字节，云服务的实际物理分配未知。 [Windows 终态原生清点的 Mac 核验](windows-final-storage.json)
