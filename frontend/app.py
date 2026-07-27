import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

API_URL = "http://localhost:8010"

st.set_page_config(
    page_title="OmniRAG",
    layout="wide",
    page_icon="\u25c6",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 960px; }
    .masthead { border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 1rem; margin-bottom: 1.5rem; }
    .masthead h1 { font-size: 1.55rem; font-weight: 650; letter-spacing: -0.02em; margin: 0; line-height: 1.1; }
    .masthead .sub { font-size: 0.78rem; color: rgba(255,255,255,0.42); margin-top: 0.35rem; }
    .strat-line { display: flex; align-items: center; gap: 0.45rem; font-size: 0.74rem; color: rgba(255,255,255,0.5); margin-bottom: 0.4rem; }
    .strat-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
    .strat-name { color: rgba(255,255,255,0.8); font-weight: 550; }
    .strat-reason { color: rgba(255,255,255,0.4); }
    .answer-meta { margin-top: 0.75rem; font-size: 0.73rem; color: rgba(255,255,255,0.38); display: flex; gap: 1.1rem; flex-wrap: wrap; }
    .answer-meta b { color: rgba(255,255,255,0.62); font-weight: 600; }
    .eyebrow { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(255,255,255,0.38); font-weight: 600; margin: 1.6rem 0 0.5rem 0; }
    button[data-baseweb="tab"] { font-size: 0.9rem; }
    .health-row { font-size: 0.8rem; line-height: 1.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="masthead">
      <h1>OmniRAG</h1>
      <div class="sub">Adaptive retrieval \u2014 routing chooses hybrid, graph, or agentic per question</div>
    </div>
    """,
    unsafe_allow_html=True,
)

STRAT_COLOR = {"hybrid": "#3B9E7A", "graph": "#5B6CC4", "agentic": "#C08A2E", "direct_llm": "#B4544A"}
STRAT_LABEL = {"hybrid": "Hybrid retrieval", "graph": "Graph retrieval", "agentic": "Agentic retrieval", "direct_llm": "Direct answer"}

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.markdown("**Retrieval**")
    strategy = st.selectbox("Strategy", ["auto", "hybrid", "graph", "agentic"],
                            help="auto lets the router pick per question", label_visibility="collapsed")
    el_filter = st.selectbox("Content type", ["all", "text", "table", "image"])
    # Document scope — restrict a query to one uploaded file so a large
    # document can't drown out a small one in retrieval.
    try:
        _docs = requests.get(f"{API_URL}/documents", timeout=5).json().get("documents", [])
    except Exception:
        _docs = []
    doc_scope = st.selectbox("Document", ["All documents"] + _docs)

    st.divider()
    st.markdown("**Add a document**")
    uploaded = st.file_uploader("Upload",
        type=["pdf", "txt", "md", "docx", "xlsx", "csv", "pptx", "json", "jsonl",
              "html", "eml", "epub", "py", "js", "ts", "java", "go", "rst", "xml"],
        label_visibility="collapsed")
    if uploaded and st.button("Index this file", type="primary", use_container_width=True):
        with st.spinner(f"Indexing {uploaded.name}"):
            resp = requests.post(f"{API_URL}/ingest/file",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")})
        if resp.status_code == 200:
            st.success(f"Indexed {uploaded.name}")
        else:
            st.error(f"Couldn't index it ({resp.status_code}).")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("**Status**")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        q = "\u25cf" if health.get("qdrant") else "\u25cb"
        m = "\u25cf" if health.get("models_loaded") else "\u25cb"
        qc = "#3B9E7A" if health.get("qdrant") else "#B4544A"
        mc = "#3B9E7A" if health.get("models_loaded") else "#B4544A"
        st.markdown(
            f"""<div class="health-row">
            <span style="color:{qc}">{q}</span> Vector store<br>
            <span style="color:{mc}">{m}</span> Models loaded<br>
            <span style="color:rgba(255,255,255,0.4)">Cache {health.get('cache_hit_rate', 0):.0%} hit rate</span>
            </div>""", unsafe_allow_html=True)
    except Exception:
        st.caption("API offline. Start it with `uvicorn api.main:app --port 8010`")

tab1, tab2, tab3 = st.tabs(["Ask", "Quality", "Pipeline"])

with tab1:
    def render_meta(data):
        n_src = len([s for s in data.get("sources", []) if s])
        latency_s = data.get("latency_ms", 0) / 1000
        cache = "from cache" if data.get("cache_hit") else f"{latency_s:.1f}s"
        st.markdown(
            f"""<div class="answer-meta">
                <span><b>{n_src}</b> source{'s' if n_src != 1 else ''}</span>
                <span><b>{data.get('chunks_retrieved', 0)}</b> chunks</span>
                <span>{cache}</span>
            </div>""", unsafe_allow_html=True)
        srcs = [s for s in data.get("sources", []) if s]
        if srcs:
            with st.expander(f"Sources ({len(srcs)})"):
                for s in srcs:
                    st.markdown(f"\u2014 {s}")

    def render_strat(data):
        strat = data.get("strategy", "hybrid")
        color = STRAT_COLOR.get(strat, "#8A8A8A")
        label = STRAT_LABEL.get(strat, strat)
        reason = data.get("strategy_reason", "")
        st.markdown(
            f"""<div class="strat-line">
                <span class="strat-dot" style="background:{color}"></span>
                <span class="strat-name">{label}</span>
                <span class="strat-reason">\u2014 {reason}</span>
            </div>""", unsafe_allow_html=True)

    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                data = msg.get("meta", {})
                if data:
                    render_strat(data)
                st.markdown(msg["content"])
                if data:
                    render_meta(data)

    prompt = st.chat_input("Ask a question about your documents\u2026")

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Working through it\u2026"):
                payload = {"query": prompt, "strategy": strategy}
                if el_filter != "all":
                    payload["element_type_filter"] = el_filter
                if doc_scope and doc_scope != "All documents":
                    payload["source_filter"] = doc_scope
                try:
                    resp = requests.post(f"{API_URL}/query", json=payload, timeout=120)
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the API. Start it with `uvicorn api.main:app --port 8010`")
                    st.stop()

            if resp.status_code == 200:
                data = resp.json()
                st.session_state["last_query"] = data
                render_strat(data)
                st.markdown(data["answer"])
                render_meta(data)
                st.session_state["messages"].append({"role": "assistant", "content": data["answer"], "meta": data})
            else:
                err = f"The request failed ({resp.status_code}). {resp.text}"
                st.error(err)
                st.session_state["messages"].append({"role": "assistant", "content": err, "meta": {}})

with tab2:
    try:
        stats = requests.get(f"{API_URL}/eval/stats", timeout=5).json()
        history = requests.get(f"{API_URL}/eval/history", timeout=5).json()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Faithfulness", f"{stats.get('avg_faithfulness', 0):.2f}")
        c2.metric("Relevancy", f"{stats.get('avg_answer_relevancy', 0):.2f}")
        c3.metric("Precision", f"{stats.get('avg_context_precision', 0):.2f}")
        c4.metric("Avg latency", f"{stats.get('avg_latency_ms', 0):.0f} ms")
        c5.metric("Queries", int(stats.get("total_queries", 0)))
        if history:
            df = pd.DataFrame(history)
            if len(df) > 1:
                st.markdown('<div class="eyebrow">Scores over time</div>', unsafe_allow_html=True)
                fig = px.line(df.tail(50), x="timestamp",
                              y=["faithfulness", "answer_relevancy", "context_precision"],
                              labels={"value": "Score", "variable": ""},
                              color_discrete_map={"faithfulness": "#3B9E7A", "answer_relevancy": "#5B6CC4", "context_precision": "#C08A2E"})
                fig.update_layout(height=300, legend_title="", margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)
            try:
                breakdown = requests.get(f"{API_URL}/observability/strategy-breakdown", timeout=5).json()
                if breakdown:
                    df_b = pd.DataFrame(breakdown)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="eyebrow">How questions were routed</div>', unsafe_allow_html=True)
                        fig_pie = px.pie(df_b, values="count", names="strategy_used",
                                         color_discrete_sequence=["#3B9E7A", "#5B6CC4", "#C08A2E", "#B4544A"], hole=0.55)
                        fig_pie.update_layout(height=280, margin=dict(t=10))
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        st.markdown('<div class="eyebrow">Faithfulness by strategy</div>', unsafe_allow_html=True)
                        fig_bar = px.bar(df_b, x="strategy_used", y="avg_faithfulness", color="strategy_used",
                                         color_discrete_sequence=["#3B9E7A", "#5B6CC4", "#C08A2E", "#B4544A"])
                        fig_bar.update_layout(height=280, showlegend=False, margin=dict(t=10), xaxis_title="", yaxis_title="")
                        st.plotly_chart(fig_bar, use_container_width=True)
            except Exception:
                pass
            st.markdown('<div class="eyebrow">Every scored query</div>', unsafe_allow_html=True)
            cols = ["timestamp", "query", "strategy", "faithfulness", "answer_relevancy", "context_precision", "latency_ms"]
            avail = [c for c in cols if c in df.columns]
            st.dataframe(df[avail].tail(30), use_container_width=True, hide_index=True)
        else:
            st.caption("No scored queries yet.")
    except Exception as e:
        st.caption(f"Quality scores appear here once the API is running. ({e})")

with tab3:
    last = st.session_state.get("last_query")
    if last:
        st.markdown('<div class="eyebrow">Your last question</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:rgba(255,255,255,0.6); font-size:0.9rem; margin-bottom:0.75rem;">"{last.get("query", "")[:120]}"</div>', unsafe_allow_html=True)
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Chunks", last.get("chunks_retrieved", 0))
        b2.metric("Queries expanded", last.get("num_queries_expanded", 1))
        b3.metric("Cache", "Hit" if last.get("cache_hit") else "Miss")
        b4.metric("Fallback", "Yes" if last.get("fallback_triggered") else "No")
        comp = last.get("compression_ratio")
        b5.metric("Compression", f"{comp:.0%} kept" if comp else "\u2014")
        scores = last.get("ragas_scores", {})
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
        s2.metric("Relevancy", f"{scores.get('answer_relevancy', 0):.2f}")
        s3.metric("Precision", f"{scores.get('context_precision', 0):.2f}")
        s4.metric("Latency", f"{last.get('latency_ms', 0):.0f} ms")
        st.divider()
    try:
        obs = requests.get(f"{API_URL}/observability/dashboard", timeout=5).json()
        traces = requests.get(f"{API_URL}/observability/traces?limit=50", timeout=5).json()
        trend = requests.get(f"{API_URL}/observability/latency-trend?limit=30", timeout=5).json()
        st.markdown('<div class="eyebrow">Across all queries</div>', unsafe_allow_html=True)
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Avg latency", f"{obs.get('avg_total_ms', 0):.0f} ms")
        h2.metric("Cache hit rate", f"{obs.get('cache_hit_rate', 0):.0f}%")
        h3.metric("Fallback rate", f"{obs.get('fallback_rate', 0):.0f}%")
        h4.metric("Avg rerank score", f"{obs.get('avg_top_rerank_score', 0):.3f}")
        h5.metric("Total queries", int(obs.get("total_queries", 0)))
        st.markdown('<div class="eyebrow">Where the time goes</div>', unsafe_allow_html=True)
        stage_data = pd.DataFrame({
            "Stage": ["Route", "Expand", "Retrieve", "Rerank", "Compress", "Generate"],
            "ms": [obs.get("avg_route_ms", 0), obs.get("avg_expand_ms", 0), obs.get("avg_retrieve_ms", 0),
                   obs.get("avg_rerank_ms", 0), obs.get("avg_compress_ms", 0), obs.get("avg_generate_ms", 0)]})
        fig_stages = px.bar(stage_data, x="Stage", y="ms", text="ms")
        fig_stages.update_traces(texttemplate="%{text:.0f}ms", textposition="outside", marker_color="#5B6CC4")
        fig_stages.update_layout(height=300, margin=dict(t=10), xaxis_title="", yaxis_title="ms")
        st.plotly_chart(fig_stages, use_container_width=True)
        if trend:
            st.markdown('<div class="eyebrow">Latency per query</div>', unsafe_allow_html=True)
            df_trend = pd.DataFrame(trend)
            fig_trend = px.bar(df_trend, x="timestamp",
                               y=["latency_retrieve_ms", "latency_rerank_ms", "latency_compress_ms", "latency_generate_ms"],
                               labels={"value": "ms", "variable": "Stage"}, barmode="stack",
                               color_discrete_map={"latency_retrieve_ms": "#3B9E7A", "latency_rerank_ms": "#C08A2E",
                                                   "latency_compress_ms": "#B4544A", "latency_generate_ms": "#5B6CC4"})
            fig_trend.update_layout(height=300, legend_title="", margin=dict(t=10), xaxis_title="")
            st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('<div class="eyebrow">Cache</div>', unsafe_allow_html=True)
        cache_info = obs.get("cache", {})
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Hit rate", f"{cache_info.get('hit_rate', 0):.0%}")
        cb.metric("Size", f"{cache_info.get('size', 0)} / {cache_info.get('max_size', 500)}")
        cc.metric("Hits", cache_info.get("hits", 0))
        cd.metric("Evictions", cache_info.get("evictions", 0))
        st.markdown('<div class="eyebrow">Recent queries</div>', unsafe_allow_html=True)
        if traces:
            df_t = pd.DataFrame(traces)
            cols = ["timestamp", "query", "strategy_used", "cache_hit", "fallback_triggered",
                    "chunks_retrieved", "top_rerank_score", "compression_ratio", "latency_total_ms", "ragas_faithfulness"]
            avail = [c for c in cols if c in df_t.columns]
            st.dataframe(df_t[avail].head(30), use_container_width=True, hide_index=True)
        else:
            st.caption("No queries recorded yet.")
    except Exception as e:
        st.caption(f"Pipeline data appears here once the API is running. ({e})")