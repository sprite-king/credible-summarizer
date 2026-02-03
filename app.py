import csv
import html
import importlib.metadata
import json
from io import BytesIO, StringIO
from datetime import datetime, timezone
from typing import Dict, List, Optional

import streamlit as st

from credible_summarizer.i18n import t
from credible_summarizer.io_utils import (
    Document,
    PdfReader,
    export_csv,
    export_docx,
    export_experiment_record,
    export_html,
    export_json,
    export_markdown,
    extract_text_from_upload,
)
from credible_summarizer.sample import SAMPLE_TEXT
from credible_summarizer.text_utils import (
    deduplicate_indices,
    extract_keywords,
    get_context,
    has_jieba,
    highlight_text,
    map_citations,
    suggest_thresholds,
    textrank_summary,
    split_sentences,
    format_score,
)


st.set_page_config(page_title="可信摘要 / 可追溯摘要 Demo", layout="wide")

st.markdown(
    """
    <style>
    mark { background-color: #ffe066; padding: 0 2px; border-radius: 3px; }
    .origin-sent { padding: 2px 4px; border-radius: 4px; }
    .origin-hit { background-color: #e7f5ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_versions() -> Dict[str, str]:
    packages = ["streamlit", "numpy", "scikit-learn", "PyPDF2", "python-docx", "jieba"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            continue
    return versions


def apply_style(style: str, top_k: int, threshold: float, gap: float) -> tuple[int, float, float]:
    if style == "conservative":
        top_k = max(3, top_k - 1)
        threshold = min(1.0, threshold + 0.08)
        gap = min(0.5, gap + 0.03)
    elif style == "coverage":
        top_k = min(12, top_k + 1)
        threshold = max(0.0, threshold - 0.08)
        gap = max(0.0, gap - 0.03)
    return top_k, threshold, gap


def run_pipeline(
    text: str,
    top_k: int,
    top_n: int,
    threshold: float,
    gap: float,
    style: str,
    auto_suggest: bool,
    dedup: bool,
    dedup_threshold: float,
    use_jieba: bool,
) -> Optional[Dict]:
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return None

    suggested_threshold, suggested_gap = suggest_thresholds(sentences, use_jieba=use_jieba)
    if auto_suggest:
        threshold = suggested_threshold
        gap = suggested_gap

    top_k, threshold, gap = apply_style(style, top_k, threshold, gap)
    top_k = min(top_k, len(sentences))

    summary_indices = textrank_summary(sentences, top_k=top_k, use_jieba=use_jieba)
    if dedup:
        summary_indices = deduplicate_indices(
            sentences, summary_indices, threshold=dedup_threshold, use_jieba=use_jieba
        )
    citation_groups = map_citations(sentences, summary_indices, top_n=top_n, use_jieba=use_jieba)

    keywords = extract_keywords(sentences, top_k=12, use_jieba=use_jieba)
    uncertainty_flags = []
    for cites in citation_groups:
        top1 = cites[0].score if cites else 0.0
        top2 = cites[1].score if len(cites) > 1 else 0.0
        uncertain = (top1 < threshold) or ((top1 - top2) < gap)
        uncertainty_flags.append(bool(uncertain))

    return {
        "sentences": sentences,
        "summary_indices": summary_indices,
        "citations": citation_groups,
        "uncertainty_flags": uncertainty_flags,
        "keywords": keywords,
        "threshold": threshold,
        "gap": gap,
    }


def render_original(sentences: List[str], cited_indices: List[int]) -> None:
    cited = set(cited_indices)
    rows = []
    for idx, sent in enumerate(sentences):
        cls = "origin-sent origin-hit" if idx in cited else "origin-sent"
        rows.append(
            f"<div id=\"sent-{idx}\" class=\"{cls}\"><strong>[{idx + 1}]</strong> {html.escape(sent)}</div>"
        )
    st.markdown("\n".join(rows), unsafe_allow_html=True)


def render_summary(result: Dict, show_context: bool, lang: str) -> None:
    sentences = result["sentences"]
    summary_indices = result["summary_indices"]
    citation_groups = result["citations"]
    uncertainty_flags = result["uncertainty_flags"]
    keywords = result["keywords"]

    st.subheader(t("section", lang))
    cited_indices = []
    for i, idx in enumerate(summary_indices):
        flag = t("uncertain", lang) if uncertainty_flags[i] else t("trusted", lang)
        with st.container(border=True):
            highlighted = highlight_text(sentences[idx], keywords)
            st.markdown(f"**{flag}**  {highlighted}", unsafe_allow_html=True)
            if uncertainty_flags[i]:
                st.caption(t("uncertain_tip", lang))
            for cite in citation_groups[i]:
                cite_text = highlight_text(cite.sentence, keywords)
                cited_indices.append(cite.index)
                st.markdown(
                    f"- 引用（{format_score(cite.score)}）: {cite_text} "
                    f"<a href=\"#sent-{cite.index}\">{t('original', lang)}</a>",
                    unsafe_allow_html=True,
                )
                if show_context:
                    with st.expander(t("context_label", lang), expanded=False):
                        st.write(get_context(sentences, cite.index))

    with st.expander(t("original", lang), expanded=False):
        render_original(sentences, cited_indices)


def build_combined_markdown(reports: List[Dict]) -> str:
    chunks = ["# 可信摘要合并报告", ""]
    for report in reports:
        chunks.append(f"## {report['name']}")
        chunks.append(
            export_markdown(
                report["sentences"],
                report["summary_indices"],
                report["citations"],
                report["uncertainty_flags"],
            )
        )
        chunks.append("")
    return "\n".join(chunks).strip()


def build_combined_html(reports: List[Dict]) -> str:
    chunks = ["<h1>可信摘要合并报告</h1>"]
    for report in reports:
        chunks.append(f"<h2>{html.escape(report['name'])}</h2>")
        chunks.append(
            export_html(
                report["sentences"],
                report["summary_indices"],
                report["citations"],
                report["uncertainty_flags"],
            )
        )
    return "\n".join(chunks)


def build_combined_docx(reports: List[Dict]) -> bytes:
    if Document is None:
        return b""
    doc = Document()
    doc.add_heading("可信摘要合并报告", level=1)
    for report in reports:
        doc.add_heading(report["name"], level=2)
        for i, idx in enumerate(report["summary_indices"]):
            flag = "⚠️" if report["uncertainty_flags"][i] else "✅"
            doc.add_paragraph(f"{flag} {report['sentences'][idx]}", style="List Bullet")
            for cite in report["citations"][i]:
                doc.add_paragraph(
                    f"引用（{format_score(cite.score)}）: {cite.sentence}", style="List Bullet 2"
                )
    output = BytesIO()
    doc.save(output)
    return output.getvalue()


def build_json_dict(report: Dict) -> Dict:
    items = []
    for i, idx in enumerate(report["summary_indices"]):
        items.append(
            {
                "summary": report["sentences"][idx],
                "uncertain": bool(report["uncertainty_flags"][i]),
                "citations": [
                    {"sentence": c.sentence, "score": c.score, "index": c.index}
                    for c in report["citations"][i]
                ],
            }
        )
    return {"items": items}


def build_combined_json(reports: List[Dict]) -> str:
    payload = {
        "files": [
            {"name": report["name"], **build_json_dict(report)} for report in reports
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_combined_csv(reports: List[Dict]) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["file", "summary", "uncertain", "citation_sentence", "citation_score", "citation_index"])
    for report in reports:
        for i, idx in enumerate(report["summary_indices"]):
            for c in report["citations"][i]:
                writer.writerow(
                    [
                        report["name"],
                        report["sentences"][idx],
                        int(report["uncertainty_flags"][i]),
                        c.sentence,
                        format_score(c.score),
                        c.index,
                    ]
                )
    return output.getvalue()

lang = "zh"
with st.sidebar:
    lang = st.selectbox(
        "Language",
        options=["zh", "en"],
        index=0,
        format_func=lambda x: "中文" if x == "zh" else "English",
        key="lang_select",
    )

st.title(t("title", lang))
st.caption(t("subtitle", lang))

with st.sidebar:
    st.header(t("settings", lang))
    style = st.selectbox(
        t("style", lang),
        options=["balanced", "conservative", "coverage"],
        format_func=lambda x: {
            "balanced": t("style_bal", lang),
            "conservative": t("style_cons", lang),
            "coverage": t("style_cov", lang),
        }[x],
        key="style_select",
    )
    top_k = st.slider(t("points", lang), min_value=3, max_value=12, value=6)
    top_n = st.slider(t("citations", lang), min_value=1, max_value=3, value=2)
    auto_suggest = st.checkbox(t("auto", lang), value=True)
    threshold = st.slider(t("threshold", lang), 0.0, 1.0, 0.35, 0.01)
    gap = st.slider(t("gap", lang), 0.0, 0.5, 0.05, 0.01)
    dedup = st.checkbox(t("dedup", lang), value=True)
    dedup_threshold = st.slider(t("dedup_thr", lang), 0.5, 0.95, 0.8, 0.01)
    use_jieba = st.checkbox(t("use_jieba", lang), value=False)
    compare_mode = st.checkbox(t("compare", lang), value=False, help=t("compare_hint", lang))
    show_context = st.checkbox(t("context", lang), value=False)

    st.header(t("privacy", lang))
    if st.button(t("clear_cache", lang)):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()
        st.success(t("cache_cleared", lang))

text = st.text_area(t("input", lang), value=SAMPLE_TEXT, height=220, placeholder=t("placeholder", lang))

uploaded_files = st.file_uploader(
    t("upload", lang),
    type=["txt", "pdf", "doc", "docx"],
    help=t("upload_multi_hint", lang),
    accept_multiple_files=True,
)
if use_jieba and not has_jieba():
    st.warning(t("jieba_missing", lang))

batch_mode = uploaded_files is not None and len(uploaded_files) > 1
if uploaded_files:
    if not batch_mode:
        uploaded = uploaded_files[0]
        extracted = extract_text_from_upload(uploaded)
        if uploaded.name.lower().endswith(".doc"):
            st.warning(t("upload_doc_notice", lang))
        elif uploaded.name.lower().endswith(".pdf") and PdfReader is None:
            st.warning(t("upload_missing_pdf", lang))
        elif uploaded.name.lower().endswith(".docx") and Document is None:
            st.warning(t("upload_missing_docx", lang))
        if extracted:
            text = extracted

if st.button(t("run", lang), type="primary"):
    params = {
        "style": style,
        "top_k": top_k,
        "top_n": top_n,
        "auto_suggest": auto_suggest,
        "threshold": threshold,
        "gap": gap,
        "dedup": dedup,
        "dedup_threshold": dedup_threshold,
        "use_jieba": use_jieba,
        "compare_mode": compare_mode,
        "batch_mode": batch_mode,
    }

    if batch_mode:
        reports: List[Dict] = []
        for file in uploaded_files:
            extracted = extract_text_from_upload(file) or ""
            if file.name.lower().endswith(".doc"):
                st.warning(f"{file.name}: {t('upload_doc_notice', lang)}")
                continue
            if file.name.lower().endswith(".pdf") and PdfReader is None:
                st.warning(f"{file.name}: {t('upload_missing_pdf', lang)}")
                continue
            if file.name.lower().endswith(".docx") and Document is None:
                st.warning(f"{file.name}: {t('upload_missing_docx', lang)}")
                continue
            result = run_pipeline(
                extracted,
                top_k=top_k,
                top_n=top_n,
                threshold=threshold,
                gap=gap,
                style=style,
                auto_suggest=auto_suggest,
                dedup=dedup,
                dedup_threshold=dedup_threshold,
                use_jieba=use_jieba,
            )
            if result is None:
                st.error(f"{file.name}: {t('short', lang)}")
                continue
            result["name"] = file.name
            reports.append(result)

        if not reports:
            st.error(t("short", lang))
        else:
            st.subheader(t("batch_mode", lang))
            tabs = st.tabs([r["name"] for r in reports])
            for tab, report in zip(tabs, reports):
                with tab:
                    if compare_mode:
                        cols = st.columns(3)
                        configs = [
                            (t("compare_current", lang), style),
                            (t("compare_conservative", lang), "conservative"),
                            (t("compare_coverage", lang), "coverage"),
                        ]
                        for col, (title, cfg_style) in zip(cols, configs):
                            with col:
                                st.markdown(f"**{title}**")
                                cfg_result = run_pipeline(
                                    "\n".join(report["sentences"]),
                                    top_k=top_k,
                                    top_n=top_n,
                                    threshold=threshold,
                                    gap=gap,
                                    style=cfg_style,
                                    auto_suggest=auto_suggest,
                                    dedup=dedup,
                                    dedup_threshold=dedup_threshold,
                                    use_jieba=use_jieba,
                                )
                                if cfg_result is None:
                                    st.error(t("short", lang))
                                else:
                                    render_summary(cfg_result, show_context, lang)
                    else:
                        render_summary(report, show_context, lang)

            combined_md = build_combined_markdown(reports)
            combined_html = build_combined_html(reports)
            combined_json = build_combined_json(reports)
            combined_csv = build_combined_csv(reports)
            combined_docx = build_combined_docx(reports)

            st.download_button(
                label=t("combined_report", lang) + " (MD)",
                data=combined_md,
                file_name="credible_summary_batch.md",
                mime="text/markdown",
            )
            st.download_button(
                label=t("combined_report", lang) + " (HTML)",
                data=combined_html,
                file_name="credible_summary_batch.html",
                mime="text/html",
            )
            st.download_button(
                label=t("combined_report", lang) + " (JSON)",
                data=combined_json,
                file_name="credible_summary_batch.json",
                mime="application/json",
            )
            st.download_button(
                label=t("combined_report", lang) + " (CSV)",
                data=combined_csv,
                file_name="credible_summary_batch.csv",
                mime="text/csv",
            )
            if combined_docx:
                st.download_button(
                    label=t("combined_report", lang) + " (DOCX)",
                    data=combined_docx,
                    file_name="credible_summary_batch.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
    else:
        result = run_pipeline(
            text,
            top_k=top_k,
            top_n=top_n,
            threshold=threshold,
            gap=gap,
            style=style,
            auto_suggest=auto_suggest,
            dedup=dedup,
            dedup_threshold=dedup_threshold,
            use_jieba=use_jieba,
        )
        if result is None:
            st.error(t("short", lang))
        else:
            if compare_mode:
                cols = st.columns(3)
                configs = [
                    (t("compare_current", lang), style),
                    (t("compare_conservative", lang), "conservative"),
                    (t("compare_coverage", lang), "coverage"),
                ]
                for col, (title, cfg_style) in zip(cols, configs):
                    with col:
                        st.markdown(f"**{title}**")
                        cfg_result = run_pipeline(
                            text,
                            top_k=top_k,
                            top_n=top_n,
                            threshold=threshold,
                            gap=gap,
                            style=cfg_style,
                            auto_suggest=auto_suggest,
                            dedup=dedup,
                            dedup_threshold=dedup_threshold,
                            use_jieba=use_jieba,
                        )
                        if cfg_result is None:
                            st.error(t("short", lang))
                        else:
                            render_summary(cfg_result, show_context, lang)
            else:
                render_summary(result, show_context, lang)

            md = export_markdown(
                result["sentences"],
                result["summary_indices"],
                result["citations"],
                result["uncertainty_flags"],
            )
            st.download_button(
                label=t("export_md", lang),
                data=md,
                file_name="credible_summary.md",
                mime="text/markdown",
            )
            st.download_button(
                label=t("export_json", lang),
                data=export_json(
                    result["sentences"],
                    result["summary_indices"],
                    result["citations"],
                    result["uncertainty_flags"],
                ),
                file_name="credible_summary.json",
                mime="application/json",
            )
            st.download_button(
                label=t("export_csv", lang),
                data=export_csv(
                    result["sentences"],
                    result["summary_indices"],
                    result["citations"],
                    result["uncertainty_flags"],
                ),
                file_name="credible_summary.csv",
                mime="text/csv",
            )
            st.download_button(
                label=t("export_html", lang),
                data=export_html(
                    result["sentences"],
                    result["summary_indices"],
                    result["citations"],
                    result["uncertainty_flags"],
                ),
                file_name="credible_summary.html",
                mime="text/html",
            )
            docx_data = export_docx(
                result["sentences"],
                result["summary_indices"],
                result["citations"],
                result["uncertainty_flags"],
            )
            if docx_data:
                st.download_button(
                    label=t("export_docx", lang),
                    data=docx_data,
                    file_name="credible_summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

            record = export_experiment_record(params, get_versions())
            st.download_button(
                label=t("export_record", lang),
                data=record,
                file_name="credible_summary_record.json",
                mime="application/json",
            )
else:
    st.info(t("info", lang))
