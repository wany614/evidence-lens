# Evidence Lens research log

The live research log records the historical decisions through RL-016. Wany initiated the project, asked how to judge trustworthiness, and set the progression from a single image to multi-image evidence, genuine fine-tuning, a CPU implementation and continuous use. ChatGPT developed experimental protocols and implementation prompts. Codex conducted the recorded development, experiments, recovery and reproduction. This log adds events as their evidence becomes available.

## RL-017

RL-017　冻结 T1 开始执行，沿用原风险权重

事件时间为 2026 年 11 月 24 日 04:04–04:11 UTC，记录于同日。Wany 要求连续完成通用阶段，并把停止位置定在新的风险训练和领域专精之前。具体方法由 ChatGPT 在 v1.4 整理，Codex 在 Mac 上核对提交、封存权限，向既有 Windows 项目下发六份两源重建工单。

本次入口从原 G0 记录起步。六个对象为 set118、76、45、103、61、32，最初只允许每个物体的 24、22 号源机位。开工前历史账本中 T1 照片回执为零。候选照片要等同一物体的五项选择全部封存；独立目标要等 42 条条件及固定布局 288 项评分或终态失败就绪后统一开放。两份 C6 经验阈值继续为 null。

公开 v0.3.0-rc.1 四模型 CPU 包已用匿名请求重新下载。ZIP 为 164,854,827 字节，整包及内部文件已校验。这一步证明原始资产可取得。工具的新版推理、性能和跨端验收将在各自完成后记入。Claude Opus xhigh 已按传入的规格审阅检查范围，未访问机器、原图或执行代码。

来源为固定提交 cdf2267a8775030cf26616aeaa472f38ec5b2f7f、G0 入口 20261124T040929244925Z-29ed9ca73ca94672，以及匿名权重回执 20261124T041052425868Z-d308c134e69d4d4b。
