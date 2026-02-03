# 可信摘要 / 可追溯摘要小工具

一个轻量级的证据驱动摘要 Demo：抽取式摘要 + 引用证据 + 不确定提示，支持中/英界面与文件上传。

## 功能亮点
- 抽取式摘要（TextRank/TF‑IDF 图排序）
- 每条要点附带 1–2 条原文引用（证据）
- 证据不足/引用不明确自动标记 ⚠️不确定
- 关键词高亮（要点与引用句）
- 自动建议证据阈值
- 摘要风格：更保守 / 更覆盖
- 摘要对比：不同风格/阈值并排对比
- 去重（相似要点过滤）
- 一键导出 Markdown / JSON / CSV
- 模板导出：HTML / DOCX
- 批量文件处理：多文件上传与合并报告
- 引用定位跳转：点击引用定位原文
- 可复现实验记录：导出参数 + 版本 + 时间戳
- 中文分词优化：可选 jieba
- 隐私模式：一键清除本地缓存
- 文件上传：txt / pdf / docx（doc 需先转换）

## 快速开始
### 1) 安装依赖
```bash
pip install -r requirements.txt
```

### 2) 启动应用
```bash
streamlit run app.py
```

## 使用方式
1. 粘贴长文本或上传文件（txt/pdf/docx）。
2. 调整侧边栏参数（摘要条数、引用数、阈值、风格等）。
3. 点击“生成可信摘要”。
4. 导出 Markdown / JSON / CSV。

## 目录结构
```
credible-summarizer/
  app.py
  requirements.txt
  credible_summarizer/
    __init__.py
    i18n.py          # 多语言文案
    io_utils.py      # 文件读取与导出
    models.py        # 数据结构
    sample.py        # 示例文本
    text_utils.py    # 分句/摘要/相似度/高亮等核心逻辑
```

## 文件解析说明
- PDF 解析依赖：PyPDF2
- DOCX 解析依赖：python-docx
- DOC 请先转换为 DOCX 或 TXT

## 原理简述
- 先对文本分句
- 使用字符 n‑gram TF‑IDF 计算相似度图
- TextRank 选取 Top‑K 作为摘要要点
- 要点与原文句相似度 Top‑N 作为引用证据
- 相似度阈值与差距阈值判断不确定性
