# src/eval_dashboard.py
# Run with: streamlit run src/eval_dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title = 'TeleRAG Evaluation Dashboard',
    page_icon  = '📊',
    layout     = 'wide',
)

st.title('📊 TeleRAG — KPI Evaluation Dashboard')
st.divider()

KPI_TARGETS = {
    'mrr':               0.75,
    'topk_accuracy':     0.85,
    'accuracy':          0.80,
    'faithfulness':      0.90,
    'context_recall':    0.85,
    'context_precision': 0.75,
}

METRIC_LABELS = {
    'mrr':               'MRR',
    'topk_accuracy':     'Top-k Accuracy',
    'accuracy':          'Accuracy',
    'faithfulness':      'Faithfulness',
    'context_recall':    'Context Recall',
    'context_precision': 'Context Precision',
}

# ── Load results ──────────────────────────────────────────────
results_path = Path('evaluation_results.csv')
if not results_path.exists():
    st.warning(
        'No evaluation_results.csv found.\n\n'
        'Run `python src/evaluate.py` first, then reload this page.'
    )
    st.stop()

df     = pd.read_csv(results_path)
scores = {m: float(df[m].iloc[0]) for m in KPI_TARGETS if m in df.columns}

# ── Summary metrics ───────────────────────────────────────────
st.subheader('KPI Summary')
cols = st.columns(6)
for i, (metric, target) in enumerate(KPI_TARGETS.items()):
    value  = scores.get(metric, 0.0)
    passed = value >= target
    delta  = f'{value - target:+.3f} vs target'
    cols[i].metric(
        label       = METRIC_LABELS[metric],
        value       = f'{value:.3f}',
        delta       = delta,
        delta_color = 'normal' if passed else 'inverse',
    )

st.divider()

# ── Bar chart + Pie chart ─────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('Score vs Target')
    metrics = list(KPI_TARGETS.keys())
    vals    = [scores.get(m, 0.0) for m in metrics]
    targets = [KPI_TARGETS[m]     for m in metrics]
    colors  = ['#22c55e' if v >= t else '#ef4444'
               for v, t in zip(vals, targets)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x            = [METRIC_LABELS[m] for m in metrics],
        y            = vals,
        name         = 'Score',
        marker_color = colors,
        text         = [f'{v:.3f}' for v in vals],
        textposition = 'outside',
    ))
    fig.add_trace(go.Scatter(
        x      = [METRIC_LABELS[m] for m in metrics],
        y      = targets,
        name   = 'Target',
        mode   = 'markers',
        marker = dict(
            symbol = 'line-ew',
            size   = 20,
            color  = 'white',
            line   = dict(width=2, color='white'),
        ),
    ))
    fig.update_layout(
        yaxis_range   = [0, 1.15],
        plot_bgcolor  = '#0f172a',
        paper_bgcolor = '#0f172a',
        font_color    = 'white',
        height        = 400,
        showlegend    = True,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader('Pass / Fail')
    passed = sum(1 for m, t in KPI_TARGETS.items()
                 if scores.get(m, 0) >= t)
    failed = len(KPI_TARGETS) - passed

    fig2 = go.Figure(go.Pie(
        labels        = ['PASS', 'FAIL'],
        values        = [passed, failed],
        marker_colors = ['#22c55e', '#ef4444'],
        hole          = 0.55,
        textinfo      = 'label+value',
    ))
    fig2.update_layout(
        paper_bgcolor = '#0f172a',
        font_color    = 'white',
        height        = 400,
        annotations   = [dict(
            text       = f'{passed}/{len(KPI_TARGETS)}',
            font_size  = 28,
            showarrow  = False,
            font_color = 'white',
        )],
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Radar chart ───────────────────────────────────────────────
st.subheader('Radar Overview')
metric_keys    = list(KPI_TARGETS.keys())
score_vals     = [scores.get(m, 0.0) for m in metric_keys]
target_vals    = [KPI_TARGETS[m]     for m in metric_keys]
metric_names   = [METRIC_LABELS[m]   for m in metric_keys]

# Close the radar polygon
score_vals_c  = score_vals  + [score_vals[0]]
target_vals_c = target_vals + [target_vals[0]]
names_c       = metric_names + [metric_names[0]]

fig3 = go.Figure()
fig3.add_trace(go.Scatterpolar(
    r          = score_vals_c,
    theta      = names_c,
    fill       = 'toself',
    name       = 'TeleRAG',
    line_color = '#3b82f6',
    fillcolor  = 'rgba(59,130,246,0.2)',
))
fig3.add_trace(go.Scatterpolar(
    r          = target_vals_c,
    theta      = names_c,
    fill       = 'toself',
    name       = 'Target',
    line_color = '#f59e0b',
    fillcolor  = 'rgba(245,158,11,0.1)',
    line_dash  = 'dash',
))
fig3.update_layout(
    polar = dict(
        radialaxis = dict(visible=True, range=[0, 1]),
        bgcolor    = '#1e293b',
    ),
    paper_bgcolor = '#0f172a',
    font_color    = 'white',
    height        = 480,
    showlegend    = True,
)
st.plotly_chart(fig3, use_container_width=True)

# ── Raw data ──────────────────────────────────────────────────
with st.expander('📋 Raw evaluation data'):
    st.dataframe(df)
    st.download_button(
        label     = '⬇️ Download CSV',
        data      = df.to_csv(index=False),
        file_name = 'telerag_evaluation.csv',
        mime      = 'text/csv',
    )