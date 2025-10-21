import os, io, base64, math, time, itertools, random
import pandas as pd
import numpy as np
import networkx as nx
from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go

# ====================== CONFIG ======================
CSV_PATH = "studies.csv"
SEPARATOR = ";"                 # split multiple indicators in a cell
EDGE_MIN_SHARED_DEFAULT = 1     # min shared indicators to draw an edge
EDGE_JACCARD_MIN_DEFAULT = 0.2  # min Jaccard similarity for an edge
RANDOM_SEED = 42
MAX_LABEL_LEN = 90
# ====================================================

random.seed(RANDOM_SEED)
np.random.seed(0)

CANON = ["id","title","year","domain","indicators","context","design","measures","outcomes","quality","link"]

def _safe_str(x):
    if pd.isna(x): return ""
    s = str(x)
    return "" if s.strip().lower() == "nan" else s

def _split_inds(s):
    if not s: return []
    return [t.strip().lower() for t in str(s).split(SEPARATOR) if t.strip()]

def read_csv_latest(path=CSV_PATH):
    # Try UTF-8 first; fall back to latin-1 to avoid decode errors
    for enc in ["utf-8", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        # If no file yet, create an empty frame with required columns
        df = pd.DataFrame(columns=CANON)

    # Ensure required columns exist
    for c in CANON:
        if c not in df.columns:
            df[c] = ""

    # Clean/normalize
    for c in CANON:
        if c == "year": continue
        df[c] = df[c].map(_safe_str)

    # Robust year
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Indicators list
    df["indicators_list"] = df["indicators"].map(_split_inds)

    # Drop completely empty rows (no title and no indicators and no link)
    keep = (~(df["title"].str.strip()=="") | df["indicators_list"].map(bool) | ~(df["link"].str.strip()==""))
    df = df[keep].copy()

    # Stable id if missing
    df["id"] = df.apply(lambda r: r["id"] if r["id"] else f"S_{hash((r['title'], r['year'], r['domain'])) & 0xffffffff:x}", axis=1)

    return df.reset_index(drop=True)

def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B: return 0.0
    u = len(A|B)
    i = len(A&B)
    return (i/u) if u else 0.0

def build_graph(sub, min_shared, min_jaccard):
    G = nx.Graph()
    # Add nodes
    for _, r in sub.iterrows():
        G.add_node(
            r["id"],
            title=_safe_str(r["title"]),
            year=int(r["year"]) if pd.notna(r["year"]) else None,
            domain=_safe_str(r["domain"]) or "Unknown",
            indicators=r["indicators_list"],
            context=_safe_str(r["context"]),
            design=_safe_str(r["design"]),
            measures=_safe_str(r["measures"]),
            outcomes=_safe_str(r["outcomes"]),
            quality=_safe_str(r["quality"]),
            link=_safe_str(r["link"]),
        )
    # Add edges
    rows = list(sub.itertuples(index=False))
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            A, B = set(rows[i].indicators_list), set(rows[j].indicators_list)
            shared = len(A & B)
            if shared >= min_shared:
                sim = jaccard(A, B)
                if sim >= min_jaccard:
                    G.add_edge(rows[i].id, rows[j].id, shared=shared, jaccard=sim)
    return G

def layout(G):
    if len(G) == 0:
        return {}
    return nx.spring_layout(G, seed=RANDOM_SEED, iterations=200, weight="jaccard")

def fig_from(G, pos):
    if len(G) == 0:
        return go.Figure().update_layout(
            template="plotly_white",
            title="No data / no nodes — add rows to studies.csv and click Refresh CSV",
            margin=dict(l=10, r=10, t=40, b=10)
        )

    # Domain palette (numeric)
    domains = sorted({G.nodes[n]["domain"] for n in G.nodes})
    dom_index = {d:i for i,d in enumerate(domains)}

    # Edge trace
    edge_x, edge_y, edge_w = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_w.append(d.get("jaccard", 0.2))
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1), opacity=0.4, hoverinfo="none"
    )

    # Node trace
    order = list(G.nodes)
    node_x = [pos[n][0] for n in order]
    node_y = [pos[n][1] for n in order]
    sizes  = [10 + 2*len(G.nodes[n].get("indicators", [])) for n in order]
    colors = [dom_index[G.nodes[n]["domain"]] for n in order]
    hover_texts = []
    for n in order:
        nd = G.nodes[n]
        title = _safe_str(nd["title"])
        short = title if len(title) <= MAX_LABEL_LEN else title[:MAX_LABEL_LEN-1] + "…"
        inds = ", ".join(nd["indicators"][:6])
        link = nd["link"] if nd["link"] else "(no link)"
        year = nd["year"] if nd["year"] is not None else "—"
        hover_texts.append(
            f"<b>{short}</b> ({year})<br>"
            f"<b>Domain:</b> {nd['domain']}<br>"
            f"<b>Indicators:</b> {inds}<br>"
            f"<b>Link:</b> {link}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Domain", tickvals=list(range(len(domains))), ticktext=domains),
            line=dict(width=0.5, color="#333")
        ),
        hovertemplate="%{text}<extra></extra>",
        text=hover_texts
    )
    node_trace.customdata = [G.nodes[n]["link"] for n in order]

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        hovermode="closest",
        dragmode="pan",
        title="Drivers of Disengagement — Interactive Map"
    )
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig

def unique_sorted(seq):
    return sorted([s for s in set(seq) if _safe_str(s)])

def vocab_indicators(df):
    bag = set()
    for L in df["indicators_list"]:
        for t in L:
            bag.add(t)
    return sorted(bag)

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Drivers of Disengagement — Interactive Map"),

    # invisible version store to trigger UI refresh after CSV upload
    dcc.Store(id="ver"),

    html.Div([
        dcc.Upload(
            id="upload",
            children=html.Div(["Drag & Drop or ", html.B("Select CSV")]),
            multiple=False,
            style={
                "width":"240px","height":"40px","lineHeight":"40px",
                "borderWidth":"1px","borderStyle":"dashed","borderRadius":"6px",
                "textAlign":"center","marginRight":"10px","display":"inline-block"
            }
        ),
        html.Button("Refresh CSV", id="refresh", n_clicks=0, style={"marginRight":"10px"}),
        html.Span(id="counts", style={"fontSize":"14px"})
    ], style={"display":"flex","alignItems":"center","gap":"8px","flexWrap":"wrap","marginBottom":"8px"}),

    html.Div([
        html.Div([
            html.Label("Domain"),
            dcc.Dropdown(id="domain", multi=True, placeholder="All domains"),
        ], style={"minWidth":"200px","flex":"1"}),

        html.Div([
            html.Label("Indicators"),
            dcc.Dropdown(id="inds", multi=True, placeholder="Any indicators"),
        ], style={"minWidth":"260px","flex":"2"}),

        html.Div([
            html.Label("Years"),
            dcc.RangeSlider(id="years", allowCross=False, tooltip={"placement":"bottom", "always_visible":False})
        ], style={"minWidth":"240px","flex":"2","padding":"0 8px"}),

        html.Div([
            html.Label("Min shared"),
            dcc.Input(id="minshared", type="number", min=0, step=1, value=EDGE_MIN_SHARED_DEFAULT, style={"width":"90px"}),
        ], style={"minWidth":"120px"}),

        html.Div([
            html.Label("Min Jaccard"),
            dcc.Input(id="minjac", type="number", min=0, max=1, step=0.05, value=EDGE_JACCARD_MIN_DEFAULT, style={"width":"110px"}),
        ], style={"minWidth":"130px"}),
    ], style={"display":"grid","gridTemplateColumns":"repeat(5,minmax(120px,1fr))","gap":"10px","alignItems":"end","marginBottom":"8px"}),

    dcc.Graph(id="graph", style={"height":"78vh","border":"1px solid #eee","borderRadius":"8px","background":"#fff"})
], style={"padding":"10px 12px","fontFamily":"-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial"})

# Upload handler: save as studies.csv and bump version to trigger UI refresh
@app.callback(
    Output("ver","data"),
    Input("upload","contents"),
    State("upload","filename"),
    State("ver","data"),
    prevent_initial_call=True
)
def on_upload(contents, filename, ver):
    if not contents:
        return no_update
    try:
        _, b64 = contents.split(",", 1)
        data = base64.b64decode(b64)
        with open(CSV_PATH, "wb") as f:
            f.write(data)
        return int(time.time())
    except Exception:
        return ver

# Update filter options (domain, indicators, year range) when CSV changes or Refresh is clicked
@app.callback(
    Output("domain","options"),
    Output("inds","options"),
    Output("years","min"),
    Output("years","max"),
    Output("years","value"),
    Input("refresh","n_clicks"),
    Input("ver","data"),
    prevent_initial_call=False
)
def refresh_filters(_n, _v):
    df = read_csv_latest()
    doms = [{"label": d, "value": d} for d in unique_sorted(df["domain"])]
    inds = [{"label": i, "value": i} for i in vocab_indicators(df)]
    # sensible year bounds
    yrs = df["year"].dropna().astype(int)
    if len(yrs) == 0:
        yrmin, yrmax = 2015, 2025
    else:
        yrmin, yrmax = int(yrs.min()), int(max(yrs.max(), yrs.min()))
        # expand to at least 2015–2025 if narrow
        yrmin = min(yrmin, 2015); yrmax = max(yrmax, 2025)
    return doms, inds, yrmin, yrmax, [yrmin, yrmax]

def apply_filters(df, doms, inds, year_range):
    if doms:
        df = df[df["domain"].isin(doms)]
    if inds:
        df = df[df["indicators_list"].map(lambda L: any(i in L for i in inds))]
    if year_range and len(year_range) == 2:
        a, b = year_range
        df = df[(df["year"].isna()) | ((df["year"] >= a) & (df["year"] <= b))]
    return df

# Main graph update: always re-read CSV
@app.callback(
    Output("graph","figure"),
    Output("counts","children"),
    Input("domain","value"),
    Input("inds","value"),
    Input("years","value"),
    Input("minshared","value"),
    Input("minjac","value"),
    Input("refresh","n_clicks"),
    Input("ver","data")
)
def update_graph(doms, inds, years, minshared, minjac, _n, _v):
    df = read_csv_latest()
    minshared = int(minshared) if (minshared is not None and minshared == minshared) else EDGE_MIN_SHARED_DEFAULT
    minjac = float(minjac) if (minjac is not None and minjac == minjac) else EDGE_JACCARD_MIN_DEFAULT
    sub = apply_filters(df, doms or [], inds or [], years or [])
    G = build_graph(sub, minshared, minjac)
    pos = layout(G)
    fig = fig_from(G, pos)
    indicator_count = len({i for L in sub["indicators_list"] for i in L})
    info = f"{len(sub)} studies | {G.number_of_edges()} edges | {indicator_count} unique indicators"
    return fig, info

# Clickable nodes -> open link in new tab
app.clientside_callback(
    """
    function(clickData) {
      if (clickData && clickData.points && clickData.points.length > 0) {
        const url = clickData.points[0].customdata;
        if (url && (url.startsWith('http://') || url.startsWith('https://'))) {
          window.open(url, '_blank');
        }
      }
      return null;
    }
    """,
    Output("counts","title"),
    Input("graph","clickData"),
)

if __name__ == "__main__":
    print("✅ App ready. Open http://127.0.0.1:8066")
    # Dash 2.x uses run_server
    app.run_server(debug=False, host="127.0.0.1", port=8066)
