import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# Silver & Platinum Dashboard
# Inspired by the attached Gold dashboard structure/theme.
# =========================================================

st.set_page_config(
    page_title="Silver & Platinum Dashboards",
    page_icon="🥈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Theme system
# -------------------------
@dataclass(frozen=True)
class Theme:
    key: str
    name: str
    emoji: str
    accent: str
    accent2: str
    accent_soft: str
    bg1: str
    bg2: str
    sidebar_bg: str
    sidebar_text: str
    card_bg: str
    card_border: str
    shadow: str


THEMES: Dict[str, Theme] = {
    "silver": Theme(
        key="silver",
        name="Silver",
        emoji="🥈",
        accent="#94A3B8",       # slate-400
        accent2="#334155",      # slate-700
        accent_soft="rgba(148, 163, 184, 0.18)",
        bg1="#F8FAFC",          # slate-50
        bg2="#E0F2FE",          # sky-100
        sidebar_bg="#0B1220",
        sidebar_text="#E5E7EB",
        card_bg="rgba(255,255,255,0.82)",
        card_border="rgba(148, 163, 184, 0.28)",
        shadow="0 14px 36px rgba(2, 6, 23, 0.14)",
    ),
    "platinum": Theme(
        key="platinum",
        name="Platinum",
        emoji="⚪",
        accent="#6B7280",       # gray-500
        accent2="#14B8A6",      # teal-500
        accent_soft="rgba(20, 184, 166, 0.18)",
        bg1="#F7F7FB",
        bg2="#ECFEFF",          # cyan-50
        sidebar_bg="#0B1220",
        sidebar_text="#E5E7EB",
        card_bg="rgba(255,255,255,0.82)",
        card_border="rgba(20, 184, 166, 0.22)",
        shadow="0 14px 36px rgba(2, 6, 23, 0.14)",
    ),
}


def _inject_css(theme: Theme) -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --accent: {theme.accent};
            --accent2: {theme.accent2};
            --accentSoft: {theme.accent_soft};
            --bg1: {theme.bg1};
            --bg2: {theme.bg2};
            --sidebarBg: {theme.sidebar_bg};
            --sidebarText: {theme.sidebar_text};
            --cardBg: {theme.card_bg};
            --cardBorder: {theme.card_border};
            --shadow: {theme.shadow};
        }}

        /* App background */
        .stApp {{
            background: radial-gradient(1200px 800px at 10% -10%, var(--bg2), transparent 55%),
                        radial-gradient(1100px 700px at 90% 0%, rgba(148,163,184,0.20), transparent 55%),
                        linear-gradient(180deg, var(--bg1) 0%, #ffffff 55%, var(--bg1) 100%);
        }}

        /* Layout breathing room */
        .block-container {{
            padding-top: 1.1rem;
            padding-bottom: 2.0rem;
            max-width: 1350px;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--sidebarBg), #070B14);
            border-right: 1px solid rgba(148,163,184,0.18);
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--sidebarText) !important;
        }}
        section[data-testid="stSidebar"] a {{
            color: var(--sidebarText) !important;
            text-decoration: none;
        }}

        /* Remove Streamlit default menu/footer */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* Hero header */
        .hero {{
            position: relative;
            border-radius: 22px;
            padding: 18px 18px;
            background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,255,255,0.65));
            border: 1px solid var(--cardBorder);
            box-shadow: var(--shadow);
            overflow: hidden;
            margin-bottom: 14px;
        }}
        .hero:before {{
            content: "";
            position: absolute;
            inset: -40px -60px auto auto;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle at 30% 30%, var(--accentSoft), transparent 60%);
            transform: rotate(18deg);
        }}
        .hero:after {{
            content: "";
            position: absolute;
            inset: auto auto -60px -80px;
            width: 320px;
            height: 320px;
            background: radial-gradient(circle at 60% 30%, rgba(148,163,184,0.18), transparent 65%);
            transform: rotate(-12deg);
        }}
        .hero-inner {{
            position: relative;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
        }}
        .hero-title {{
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: 0.2px;
            margin: 0;
            color: #0f172a;
        }}
        .hero-sub {{
            margin-top: 4px;
            color: #475569;
            font-size: 0.95rem;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 10px;
            border-radius: 999px;
            border: 1px solid var(--cardBorder);
            background: rgba(255,255,255,0.75);
            font-weight: 700;
            color: #0f172a;
        }}


        /* KPI cards (Gold-style, compact) */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 10px 0 10px 0;
        }}
        @media (max-width: 1100px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
        @media (max-width: 700px) {{
            .kpi-grid {{ grid-template-columns: repeat(1, minmax(0, 1fr)); }}
        }}
        .kpi {{
            background: rgba(255,255,255,.90);
            border: 1px solid var(--cardBorder);
            border-radius: 16px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
            padding: 14px 14px;
            display: flex;
            gap: 12px;
            align-items: center;
            min-height: 84px;
            overflow: hidden;
            position: relative;
        }}
        .kpi:before {{
            content: "";
            position: absolute;
            inset: -40px -40px auto auto;
            width: 140px;
            height: 140px;
            background: radial-gradient(circle at 30% 30%, var(--accentSoft), transparent 60%);
            transform: rotate(18deg);
        }}
        .kpi .ico {{
            position: relative;
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(0,0,0,.04);
            border: 1px solid rgba(0,0,0,.08);
            font-size: 22px;
            flex: 0 0 auto;
        }}
        .kpi .lbl {{
            position: relative;
            font-size: 12.5px;
            color: #64748b;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 2px;
        }}
        .kpi .val {{
            position: relative;
            font-weight: 900;
            color: #0f172a;
            line-height: 1.05;
        }}
        .kpi .sub {{
            position: relative;
            margin-top: 6px;
            font-size: 0.85rem;
            color: #475569;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 520px;
        }}


        /* Section headings */
        .section-title {{
            font-weight: 800;
            font-size: 1.02rem;
            margin: 4px 0 6px 0;
            color: #0f172a;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(title: str, subtitle: str, theme: Theme, right_svg: str = "") -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-inner">
            <div>
              <div class="badge">{theme.emoji} <span>{theme.name}</span></div>
              <h1 class="hero-title">{title}</h1>
              <div class="hero-sub">{subtitle}</div>
            </div>
            <div style="min-width:120px; opacity:0.9">{right_svg}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _svg_coin(symbol: str, theme: Theme) -> str:
    # Minimal inline SVG (keeps file self-contained)
    return f"""
    <svg width="130" height="86" viewBox="0 0 130 86" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="130" y2="86" gradientUnits="userSpaceOnUse">
          <stop stop-color="{theme.accent}"/>
          <stop offset="1" stop-color="{theme.accent2}"/>
        </linearGradient>
      </defs>
      <rect x="10" y="18" width="110" height="56" rx="18" fill="url(#g)" opacity="0.22"/>
      <circle cx="56" cy="44" r="20" fill="url(#g)" opacity="0.55"/>
      <circle cx="56" cy="44" r="18" stroke="{theme.accent2}" stroke-width="2" opacity="0.65"/>
      <text x="56" y="50" text-anchor="middle" font-size="16" font-family="Inter, system-ui, -apple-system" font-weight="800" fill="{theme.accent2}">
        {symbol}
      </text>
      <rect x="78" y="28" width="40" height="32" rx="12" stroke="{theme.accent2}" stroke-width="2" opacity="0.45"/>
    </svg>
    """


# -------------------------
# Formatting helpers
# -------------------------
def _fmt_num(x: float, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        if decimals == 0:
            return f"{x:,.0f}"
        return f"{x:,.{decimals}f}"
    except Exception:
        return "—"


def _fmt_pct(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{x:+.{decimals}f}%"
    except Exception:
        return "—"



def _add_line_point_labels(fig, fmt: str = "{:,.2f}"):
    """Add value labels on all points of each scatter/line trace."""
    for tr in getattr(fig, "data", []) or []:
        if not hasattr(tr, "y") or tr.y is None:
            continue
        try:
            yvals = list(tr.y)
        except Exception:
            yvals = [tr.y]

        if len(yvals) == 0:
            continue

        txt = []
        for v in yvals:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                txt.append("")
            else:
                try:
                    txt.append(fmt.format(float(v)))
                except Exception:
                    txt.append(str(v))

        tr.text = txt
        tr.textposition = "top center"
        mode = getattr(tr, "mode", "") or ""
        if "text" not in mode:
            tr.mode = (mode + "+text") if mode else "lines+markers+text"


def _add_bar_labels(fig, orientation: str = "v", fmt: str = "{:,.2f}"):
    """Add value labels to Plotly bar traces."""
    for tr in getattr(fig, "data", []) or []:
        if tr.type != "bar":
            continue

        vals = tr.x if orientation == "h" else tr.y
        try:
            vlist = list(vals) if vals is not None else []
        except Exception:
            vlist = [vals]

        txt = []
        for v in vlist:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                txt.append("")
            else:
                try:
                    txt.append(fmt.format(float(v)))
                except Exception:
                    txt.append(str(v))

        tr.text = txt
        tr.textposition = "outside"

def _autofont_size(value_str: str) -> str:
    # Approx. responsive font sizing for KPI values
    n = len(str(value_str))
    if n <= 6:
        return "1.85rem"
    if n <= 9:
        return "1.55rem"
    if n <= 12:
        return "1.25rem"
    return "1.05rem"



def _kpi_html(icon: str, label: str, value: str, sub: str = "") -> str:
    fs = _autofont_size(value)
    sub_html = sub if sub else "&nbsp;"
    return (
        f"""
        <div class="kpi">
          <div class="ico">{icon}</div>
          <div style="position:relative; min-width:0;">
            <div class="lbl">{label}</div>
            <div class="val" style="font-size:{fs};">{value}</div>
            <div class="sub">{sub_html}</div>
          </div>
        </div>
        """
    ).strip()


def _kpi_grid(cards: List[Tuple[str, str, str, str]]) -> None:
    # Render all KPI cards in one HTML block (so the grid stays compact)
    html = ['<div class="kpi-grid">']
    for icon, label, value, sub in cards:
        html.append(_kpi_html(icon, label, value, sub))
    html.append('</div>')
    st.markdown("\n".join(html), unsafe_allow_html=True)



def _section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# -------------------------
# Trade helpers
# -------------------------
_YEAR_RE = re.compile(r"(19|20)\d{2}")

def _extract_year(col: str) -> Optional[int]:
    m = _YEAR_RE.search(str(col))
    if not m:
        return None
    return int(m.group())


def _tidy_trade_sheet(df: pd.DataFrame, partner_col_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Tidy a wide ITC-like table: first column=partner, other columns contain years.
    Returns long columns: partner, year, value (float).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if partner_col_hint and partner_col_hint in df.columns:
        partner_col = partner_col_hint
    else:
        partner_col = df.columns[0]
    df = df.rename(columns={partner_col: "partner"})
    # Identify year columns
    year_map = {}
    for c in df.columns[1:]:
        y = _extract_year(c)
        if y is not None:
            year_map[c] = y
    keep_cols = ["partner"] + list(year_map.keys())
    df = df[keep_cols].copy()
    df = df.melt(id_vars=["partner"], var_name="year_col", value_name="value")
    df["year"] = df["year_col"].map(year_map).astype("Int64")
    df = df.drop(columns=["year_col"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["partner"] = df["partner"].astype(str).str.strip()
    df = df.dropna(subset=["year"])
    return df


def _find_sheet(xls: pd.ExcelFile, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for name in xls.sheet_names:
            if rx.search(name):
                return name
    return None


@st.cache_data(show_spinner=False)
def load_trade_total(path: str, hs_code: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load total Imports/Exports sheets for a given HS code.
    """
    xls = pd.ExcelFile(path)
    imp_sheet = _find_sheet(
        xls,
        patterns=[
            rf"^imports.*\(?{hs_code}\)?$",
            rf"^imports\({hs_code}\)$",
            rf"^{hs_code}\(imports\)$",
            rf"imports.*{hs_code}",
            rf"{hs_code}.*imports",
        ],
    )
    exp_sheet = _find_sheet(
        xls,
        patterns=[
            rf"^exports.*\(?{hs_code}\)?$",
            rf"^exports\({hs_code}\)$",
            rf"^{hs_code}\(exports\)$",
            rf"exports.*{hs_code}",
            rf"{hs_code}.*exports",
        ],
    )
    if imp_sheet is None or exp_sheet is None:
        # Fallback: try standard "Imports(...)" / "Exports(...)" naming
        imp_sheet = imp_sheet or _find_sheet(xls, [r"imports"])
        exp_sheet = exp_sheet or _find_sheet(xls, [r"exports"])
    if imp_sheet is None or exp_sheet is None:
        raise ValueError(f"Could not locate Imports/Exports sheets for HS {hs_code} in {path}")

    df_imp_wide = pd.read_excel(path, sheet_name=imp_sheet)
    df_exp_wide = pd.read_excel(path, sheet_name=exp_sheet)
    df_imp = _tidy_trade_sheet(df_imp_wide)
    df_exp = _tidy_trade_sheet(df_exp_wide)
    return df_imp, df_exp


@st.cache_data(show_spinner=False)
def load_trade_hs6_blocks(path: str, sheet_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load ITC-style HS6 sheet that contains 'Importers' and 'Exporters' blocks, plus a product descriptor.
    Returns (imports_long, exports_long, product_desc).
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    product_desc = ""
    # Try to read descriptor from row 0 col 1
    try:
        if pd.notna(raw.iat[0, 1]):
            product_desc = str(raw.iat[0, 1]).strip()
    except Exception:
        product_desc = ""

    # Locate blocks
    col0 = raw.iloc[:, 0].astype(str)
    imp_start = col0[col0.str.strip().str.lower() == "importers"].index
    exp_start = col0[col0.str.strip().str.lower() == "exporters"].index

    if len(imp_start) == 0 or len(exp_start) == 0:
        raise ValueError(f"Sheet '{sheet_name}' does not appear to have Importers/Exporters blocks.")

    imp_i = int(imp_start[0])
    exp_i = int(exp_start[0])

    # Imports block: header row is imp_i, data starts imp_i+1 until exp_i-1
    imp_block = raw.iloc[imp_i:exp_i].copy()
    imp_block = imp_block.dropna(axis=1, how="all")
    imp_block.columns = imp_block.iloc[0]
    imp_block = imp_block.iloc[1:].copy()
    # First col should be partners
    imp_block = imp_block.rename(columns={imp_block.columns[0]: "partner"})
    imp_long = _tidy_trade_sheet(imp_block, partner_col_hint="partner")

    # Exports block: header row is exp_i, data starts exp_i+1 to end
    exp_block = raw.iloc[exp_i:].copy()
    exp_block = exp_block.dropna(axis=1, how="all")
    exp_block.columns = exp_block.iloc[0]
    exp_block = exp_block.iloc[1:].copy()
    exp_block = exp_block.rename(columns={exp_block.columns[0]: "partner"})
    exp_long = _tidy_trade_sheet(exp_block, partner_col_hint="partner")

    return imp_long, exp_long, product_desc


def _trade_unit_options() -> List[str]:
    return ["USD Thousand (Base)", "USD Million", "USD Billion"]


def _scale_trade_value(v_thousand: float, unit: str) -> float:
    if v_thousand is None or (isinstance(v_thousand, float) and np.isnan(v_thousand)):
        return np.nan
    if unit == "USD Million":
        return v_thousand / 1000.0
    if unit == "USD Billion":
        return v_thousand / 1_000_000.0
    return v_thousand


def _trade_unit_label(unit: str) -> str:
    if unit == "USD Million":
        return "USD Mn"
    if unit == "USD Billion":
        return "USD Bn"
    return "USD '000"


def _default_year_range(df: pd.DataFrame) -> Tuple[int, int]:
    years = sorted(df["year"].dropna().unique().tolist())
    if not years:
        return (2000, 2024)
    return (int(years[0]), int(years[-1]))


def _apply_year_filter(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    return df[(df["year"] >= y0) & (df["year"] <= y1)].copy()


def _plot_line(df: pd.DataFrame, y_col: str, x_col: str, color: str, show_labels: bool) -> go.Figure:
    fig = px.line(df, x=x_col, y=y_col, markers=True)
    fig.update_traces(line=dict(color=color, width=3), marker=dict(size=6, color=color))
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if show_labels:
        fig.update_traces(texttemplate="%{y:,.2f}", textposition="top center")
    return fig


def _plot_bar(df: pd.DataFrame, x: str, y: str, color: str, show_labels: bool, orientation: str = "v") -> go.Figure:
    fig = px.bar(df, x=x, y=y, orientation=orientation)
    fig.update_traces(marker_color=color)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(148,163,184,0.20)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
    )
    if show_labels:
        fig.update_traces(texttemplate="%{y:,.2f}" if orientation=="v" else "%{x:,.2f}", textposition="outside")
    return fig


# -------------------------
# Silver production (USGS-like, 2-year)
# -------------------------
@st.cache_data(show_spinner=False)
def load_silver_production(path: str, sheet: str = "Silver Production") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    # Find year cols
    year_cols = []
    for c in df.columns:
        y = _extract_year(c)
        if y is not None:
            year_cols.append((c, y))
    if not year_cols:
        raise ValueError("No year columns found in Silver Production sheet.")
    year_cols = sorted(year_cols, key=lambda t: t[1])
    # Identify reserves col (if any)
    reserves_col = None
    for c in df.columns:
        if "reserve" in c.lower():
            reserves_col = c
            break
    out = df.rename(columns={df.columns[0]: "country"}).copy()
    keep = ["country"] + [c for c, _ in year_cols] + ([reserves_col] if reserves_col else [])
    out = out[keep].copy()
    # Rename year columns to int years
    for c, y in year_cols:
        out[str(y)] = pd.to_numeric(out[c], errors="coerce")
        if c != str(y):
            out = out.drop(columns=[c])
    if reserves_col:
        out["reserves"] = pd.to_numeric(out[reserves_col], errors="coerce")
        out = out.drop(columns=[reserves_col])
    else:
        out["reserves"] = np.nan
    # Normalize country labels
    out["country"] = out["country"].astype(str).str.strip()
    return out


def render_silver_production(theme: Theme, prod_path: str, show_labels: bool) -> None:
    _inject_css(theme)
    _hero(
        title="Silver Production Dashboard",
        subtitle="Mine production snapshot (latest two years) + reserves (source file)",
        theme=theme,
        right_svg=_svg_coin("Ag", theme),
    )

    try:
        df = load_silver_production(prod_path)
    except Exception as e:
        st.error(f"Could not load Silver production data: {e}")
        st.info("Check that 'Silver_Production.xlsx' (sheet: 'Silver Production') is present and formatted.")
        return

    # Years available
    years = sorted([int(c) for c in df.columns if re.fullmatch(r"\d{4}", str(c))])
    if len(years) == 0:
        st.error("No year columns detected.")
        return
    y_latest = years[-1]
    y_prev = years[-2] if len(years) >= 2 else years[-1]

    # World row detection
    world_mask = df["country"].str.lower().str.contains("world")
    world_row = df[world_mask].head(1)
    world_total_latest = float(world_row[str(y_latest)].iloc[0]) if not world_row.empty else float(df[str(y_latest)].sum(skipna=True))
    world_total_prev = float(world_row[str(y_prev)].iloc[0]) if not world_row.empty else float(df[str(y_prev)].sum(skipna=True))

    yoy = (world_total_latest / world_total_prev - 1) * 100 if pd.notna(world_total_latest) and pd.notna(world_total_prev) and world_total_prev != 0 else np.nan

    # Top producer excluding "Other countries" and "World"
    base = df[~world_mask].copy()
    base_no_other = base[~base["country"].str.lower().str.contains("other")].copy()
    top_row = base_no_other.sort_values(str(y_latest), ascending=False).head(1)
    top_country = top_row["country"].iloc[0] if not top_row.empty else "—"
    top_val = float(top_row[str(y_latest)].iloc[0]) if not top_row.empty else np.nan

    # Reserves top (optional)
    r_top = base_no_other.sort_values("reserves", ascending=False).head(1)
    r_country = r_top["country"].iloc[0] if (not r_top.empty and pd.notna(r_top["reserves"].iloc[0])) else "—"
    r_val = float(r_top["reserves"].iloc[0]) if (not r_top.empty and pd.notna(r_top["reserves"].iloc[0])) else np.nan

    cards = [
        ("Top Producer", top_country, f"{_fmt_num(top_val)} t ({y_latest})" if pd.notna(top_val) else ""),
        ("World Total", f"{_fmt_num(world_total_latest)} t", f"{y_latest} (rounded if source says so)"),
        ("World YoY", _fmt_pct(yoy, 1), f"{y_prev} → {y_latest}"),
        ("Top Reserves", r_country, f"{_fmt_num(r_val)} t" if pd.notna(r_val) else "—"),
    ]
    _kpi_grid([
        ("🏭", "Top Producer", top_country, f"{_fmt_num(top_val)} t ({y_latest})" if pd.notna(top_val) else "—"),
        ("🌍", "World Total", f"{_fmt_num(world_total_latest)} t", f"{y_prev}→{y_latest} YoY: {_fmt_pct(yoy, 1)}"),
        ("📈", "World YoY", _fmt_pct(yoy, 1), f"{y_prev} → {y_latest}"),
        ("🪨", "Top Reserves", r_country, f"{_fmt_num(r_val)} t" if pd.notna(r_val) else "—"),
    ])

    tab1, tab2, tab3 = st.tabs(["Overview", "Countries", "Download"])

    with tab1:
        _section("Global snapshot")
        colA, colB = st.columns([1.15, 0.85], gap="large")

        with colA:
            snap = pd.DataFrame({"Year": [y_prev, y_latest], "World total (t)": [world_total_prev, world_total_latest]})
            fig = px.bar(snap, x="Year", y="World total (t)")
            fig.update_traces(marker_color=theme.accent2)
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
                xaxis=dict(showgrid=False),
            )
            if show_labels:
                fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            st.info(
                f"**Units:** tonnes (t)\n\n"
                f"**Coverage:** Country mine production for {y_prev} and {y_latest}, plus reserves (if provided)."
            )
            st.caption("Tip: Use the **Countries** tab to view Top‑N producers and the full snapshot table.")

    with tab2:
        _section("Top producers")
        top_n = st.slider("Top N producers (bar chart)", 5, 15, 10, 1)
        year_sel = st.radio("Select year", options=years, index=len(years) - 1, horizontal=True)

        d = df[~world_mask].copy()
        d["value"] = d[str(year_sel)]
        d = d.dropna(subset=["value"])
        d = d.sort_values("value", ascending=False)

        # Top-N logic WITHOUT generating an extra "Others" bar if the source already provides it.
        # Many production sources include an "Other countries" row + World total.
        other_mask = d["country"].str.lower().str.contains("other")
        d_named = d[~other_mask].copy()
        d_otherline = d[other_mask].copy()

        top = d_named.head(top_n)
        rest = d_named.iloc[top_n:]

        plot_df = top[["country", "value"]].copy()

        if len(d_otherline):
            # Keep the dataset-provided 'Other countries' (or similar) rows as-is.
            plot_df = pd.concat([plot_df, d_otherline[["country", "value"]]], ignore_index=True)
        else:
            # If the dataset doesn't provide an 'Other' row, fall back to Top-N + computed Others.
            rest_sum = rest["value"].sum() if len(rest) else 0.0
            if rest_sum > 0:
                plot_df = pd.concat([plot_df, pd.DataFrame([{"country": "Others", "value": rest_sum}])], ignore_index=True)

        fig = px.bar(plot_df, x="country", y="value")
        fig.update_traces(marker_color=theme.accent2)
        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Tonnes (t)", gridcolor="rgba(148,163,184,0.25)"),
            xaxis=dict(title="", tickangle=-20, showgrid=False),
        )
        if show_labels:
            fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        _section("Snapshot table")
        table = df.copy()
        # Make world row bold by tagging for display
        st.dataframe(table, use_container_width=True, hide_index=True)

    with tab3:
        _section("Download")
        out = df.copy()
        out.insert(1, "Unit", "Tonnes (t)")
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Silver Production CSV",
            data=csv,
            file_name="silver_production.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("If you want this export in Excel with formatting (bold totals, headers, colors), tell me and I’ll add it.")


# -------------------------
# Trade page renderer (Total + optional HS6 tabs)
# -------------------------
def _compute_trade_kpis(
    imp: pd.DataFrame,
    exp: pd.DataFrame,
    unit: str,
    y0: int,
    y1: int,
    focus_country: str = "World",
) -> Dict[str, str]:
    # Filter
    imp_f = _apply_year_filter(imp, y0, y1)
    exp_f = _apply_year_filter(exp, y0, y1)

    # Latest year
    y_latest = int(max(imp_f["year"].max(), exp_f["year"].max()))
    y_prev = y_latest - 1

    def _get(df: pd.DataFrame, partner: str, year: int) -> float:
        v = df[(df["partner"].str.lower() == partner.lower()) & (df["year"] == year)]["value"]
        return float(v.iloc[0]) if len(v) else np.nan

    world_imp_latest = _scale_trade_value(_get(imp_f, focus_country, y_latest), unit)
    world_exp_latest = _scale_trade_value(_get(exp_f, focus_country, y_latest), unit)
    world_trade_latest = world_imp_latest + world_exp_latest if pd.notna(world_imp_latest) and pd.notna(world_exp_latest) else np.nan

    imp_prev = _scale_trade_value(_get(imp_f, focus_country, y_prev), unit)
    exp_prev = _scale_trade_value(_get(exp_f, focus_country, y_prev), unit)

    imp_yoy = (world_imp_latest / imp_prev - 1) * 100 if pd.notna(world_imp_latest) and pd.notna(imp_prev) and imp_prev != 0 else np.nan
    exp_yoy = (world_exp_latest / exp_prev - 1) * 100 if pd.notna(world_exp_latest) and pd.notna(exp_prev) and exp_prev != 0 else np.nan

    return {
        "latest_year": str(y_latest),
        "imports_latest": _fmt_num(world_imp_latest, 2),
        "exports_latest": _fmt_num(world_exp_latest, 2),
        "trade_latest": _fmt_num(world_trade_latest, 2),
        "imports_yoy": _fmt_pct(imp_yoy, 1),
        "exports_yoy": _fmt_pct(exp_yoy, 1),
    }


def _topn_with_others(df: pd.DataFrame, year: int, top_n: int) -> pd.DataFrame:
    d = df[df["year"] == year].copy()
    d = d.dropna(subset=["value"])
    # Drop World
    d = d[~d["partner"].str.lower().eq("world")]
    d = d.sort_values("value", ascending=False)
    top = d.head(top_n).copy()
    others = d.iloc[top_n:]["value"].sum()
    if others > 0:
        top = pd.concat([top, pd.DataFrame([{"partner": "Others", "year": year, "value": others}])], ignore_index=True)
    # Add Total row (topN + others)
    total = top["value"].sum()
    top = pd.concat([top, pd.DataFrame([{"partner": "Total", "year": year, "value": total}])], ignore_index=True)
    return top


def render_trade_content(
    theme: Theme,
    imp: pd.DataFrame,
    exp: pd.DataFrame,
    unit: str,
    show_labels: bool,
    title_prefix: str,
) -> None:
    # =========================
    # Controls (Gold-like layout)
    # =========================
    years = sorted(set(pd.to_numeric(imp["year"], errors="coerce").dropna().astype(int).tolist() +
                       pd.to_numeric(exp["year"], errors="coerce").dropna().astype(int).tolist()))
    if not years:
        st.warning("No year values found in the trade data.")
        return

    # Prefer snapshot years where BOTH World imports and World exports exist (avoids trailing empty years)
    world_imp_years = set(
        pd.to_numeric(imp[imp["partner"].str.lower().eq("world")]["year"], errors="coerce").dropna().astype(int).tolist()
    )
    world_exp_years = set(
        pd.to_numeric(exp[exp["partner"].str.lower().eq("world")]["year"], errors="coerce").dropna().astype(int).tolist()
    )
    snap_years = sorted(world_imp_years.intersection(world_exp_years)) or years


    key_base = f"trade_{title_prefix}"

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.6, 1.0])
    with c1:
        snap_year = st.selectbox("Snapshot year", snap_years, index=len(snap_years) - 1, key=f"{key_base}_year")
    with c2:
        partner_mode = st.selectbox("Partner ranking based on", ["Exports", "Imports"], index=0, key=f"{key_base}_mode")
    with c3:
        top_n = st.slider("Top N countries", 5, 30, 10, key=f"{key_base}_topn")
    with c4:
        metric = st.radio("Metric", ["Value", "Share of World (%)"], index=0, horizontal=True, key=f"{key_base}_metric")

    ctx = (
        f"<div class='small-note'>"
        f"<b>Snapshot year:</b> {snap_year} &nbsp; • &nbsp; "
        f"<b>Partner ranking:</b> {partner_mode} &nbsp; • &nbsp; "
        f"<b>Top N:</b> {top_n} &nbsp; • &nbsp; "
        f"<b>Display unit:</b> {_trade_unit_label(unit)} (base: USD thousand)"
        f"</div>"
    )

    # =========================
    # World series helpers (base: USD thousand)
    # =========================
    def _world_series(df: pd.DataFrame) -> Dict[int, float]:
        d = df[df["partner"].str.lower().eq("world")].copy()
        if d.empty:
            return {}
        g = d.groupby("year", as_index=True)["value"].sum()
        out = {}
        for yy, vv in g.items():
            try:
                out[int(yy)] = float(vv)
            except Exception:
                continue
        return out

    def _yoy(series: Dict[int, float], year: int) -> float:
        v1 = float(series.get(year, np.nan))
        v0 = float(series.get(year - 1, np.nan))
        if np.isnan(v1) or np.isnan(v0) or v0 == 0:
            return np.nan
        return (v1 / v0 - 1) * 100.0

    def _cagr(series: Dict[int, float], start_year: int, end_year: int) -> float:
        v0 = float(series.get(start_year, np.nan))
        v1 = float(series.get(end_year, np.nan))
        n = int(end_year - start_year)
        if n <= 0 or np.isnan(v0) or np.isnan(v1) or v0 <= 0 or v1 <= 0:
            return np.nan
        return (v1 / v0) ** (1.0 / n) - 1.0

    world_exp = _world_series(exp)
    world_imp = _world_series(imp)

    exp_val_th = float(world_exp.get(snap_year, np.nan))
    imp_val_th = float(world_imp.get(snap_year, np.nan))
    bal_val_th = exp_val_th - imp_val_th if not (np.isnan(exp_val_th) or np.isnan(imp_val_th)) else np.nan

    # Display-scaled
    exp_disp = _scale_trade_value(exp_val_th, unit)
    imp_disp = _scale_trade_value(imp_val_th, unit)
    bal_disp = _scale_trade_value(bal_val_th, unit)

    exp_yoy = _yoy(world_exp, snap_year)
    imp_yoy = _yoy(world_imp, snap_year)
    exp_cagr = _cagr(world_exp, years[0], snap_year)
    imp_cagr = _cagr(world_imp, years[0], snap_year)

    # =========================
    # Tabs (Gold-like)
    # =========================
    tabs = st.tabs(["Overview", "Countries", "Country trend", "Download"])

    # ----- Overview -----
    with tabs[0]:
        st.markdown(f"## Overview • {snap_year}")
        st.markdown(ctx, unsafe_allow_html=True)


        trade_val_th = exp_val_th + imp_val_th if not (np.isnan(exp_val_th) or np.isnan(imp_val_th)) else np.nan
        trade_disp = _scale_trade_value(trade_val_th, unit)
        bal_note = (
            "Surplus" if (pd.notna(bal_val_th) and bal_val_th > 0)
            else "Deficit" if (pd.notna(bal_val_th) and bal_val_th < 0)
            else ""
        )

        _kpi_grid([
            ("📥", "World Imports", f"{_fmt_num(imp_disp, 2)} {_trade_unit_label(unit)}", f"YoY: {_fmt_pct(imp_yoy, 1)} (latest: {snap_year})"),
            ("📤", "World Exports", f"{_fmt_num(exp_disp, 2)} {_trade_unit_label(unit)}", f"YoY: {_fmt_pct(exp_yoy, 1)} (latest: {snap_year})"),
            ("🌍", "World Trade", f"{_fmt_num(trade_disp, 2)} {_trade_unit_label(unit)}", f"Imports + Exports ({snap_year})"),
            ("⚖️", "Trade Balance", f"{_fmt_num(bal_disp, 2)} {_trade_unit_label(unit)}", bal_note),
            ("📈", f"CAGR Exports ({years[0]}–{snap_year})", _fmt_pct(exp_cagr * 100 if pd.notna(exp_cagr) else np.nan, 2), ""),
            ("📉", f"CAGR Imports ({years[0]}–{snap_year})", _fmt_pct(imp_cagr * 100 if pd.notna(imp_cagr) else np.nan, 2), ""),
        ])


        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
        st.markdown(f"### Global trade trend • {years[0]}–{years[-1]}")

        df_world = pd.DataFrame({
            "year": years,
            "exports_th": [world_exp.get(y, np.nan) for y in years],
            "imports_th": [world_imp.get(y, np.nan) for y in years],
        })
        df_world["exports"] = df_world["exports_th"].apply(lambda v: _scale_trade_value(v, unit))
        df_world["imports"] = df_world["imports_th"].apply(lambda v: _scale_trade_value(v, unit))
        df_world["balance"] = df_world["exports"] - df_world["imports"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_world["year"], y=df_world["exports"], mode="lines+markers", name="Exports",
                                 line=dict(color=theme.accent, width=3)))
        fig.add_trace(go.Scatter(x=df_world["year"], y=df_world["imports"], mode="lines+markers", name="Imports",
                                 line=dict(color=theme.accent2, width=3)))
        fig.add_trace(go.Scatter(x=df_world["year"], y=df_world["balance"], mode="lines", name="Trade Balance",
                                 line=dict(color="rgba(15,23,42,.55)", width=2, dash="dot"), yaxis="y2"))
        fig.add_vline(x=snap_year, line_dash="dot", line_width=2, line_color="rgba(15,23,42,.35)")
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=60, b=10),
            title=f"Global trade trend ({_trade_unit_label(unit)})",
            legend_title_text="",
            yaxis=dict(title=_trade_unit_label(unit)),
            yaxis2=dict(title="Balance", overlaying="y", side="right", showgrid=False),
        )
        if show_labels:
            _add_line_point_labels(fig, fmt="{:,.2f}")
        st.plotly_chart(fig, use_container_width=True)

    # ----- Countries -----
    with tabs[1]:
        st.markdown(f"## Countries • {partner_mode} • {snap_year}")
        st.markdown(ctx, unsafe_allow_html=True)

        df_src = exp if partner_mode == "Exports" else imp
        # World total (base: USD thousand) used for share computations
        world_total_th = exp_val_th if partner_mode == "Exports" else imp_val_th

        rank = df_src[df_src["year"] == snap_year].copy()
        rank["value"] = pd.to_numeric(rank["value"], errors="coerce")
        rank = rank[rank["partner"].str.lower() != "world"].dropna(subset=["value"]).sort_values("value", ascending=False)

        if (world_total_th is None) or (isinstance(world_total_th, float) and np.isnan(world_total_th)) or world_total_th == 0:
            world_total_th = float(rank["value"].sum())

        rank_top = rank.head(top_n).copy()
        top_sum_th = float(rank_top["value"].sum())
        others_th = float(max(0.0, float(world_total_th) - top_sum_th))

        plot_df = rank_top[["partner", "value"]].copy().rename(columns={"value": "value_thousand"})
        if others_th > 0:
            plot_df = pd.concat([plot_df, pd.DataFrame([{"partner": "Others", "value_thousand": others_th}])], ignore_index=True)

        plot_df["value_scaled"] = plot_df["value_thousand"].apply(lambda v: _scale_trade_value(v, unit))
        plot_df["share_pct"] = (plot_df["value_thousand"] / float(world_total_th)) * 100.0 if world_total_th else np.nan

        x_col = "value_scaled" if metric == "Value" else "share_pct"
        x_title = _trade_unit_label(unit) if metric == "Value" else "Share of World (%)"
        lbl_fmt = "{:,.2f}" if metric == "Value" else "{:,.2f}%"

        plot_df_plot = plot_df.sort_values(x_col, ascending=True)

        fig2 = px.bar(plot_df_plot, x=x_col, y="partner", orientation="h", title=f"Top {top_n} countries + Others")
        fig2.update_traces(marker_color=(theme.accent if partner_mode == "Exports" else theme.accent2))
        fig2.update_layout(template="plotly_white", height=520, margin=dict(l=10, r=10, t=60, b=10))
        fig2.update_xaxes(title=x_title)
        fig2.update_yaxes(title="")
        if show_labels:
            _add_bar_labels(fig2, orientation="h", fmt=lbl_fmt)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Data (Top N + Others)")

        value_col = f"value ({_trade_unit_label(unit)})"
        tbl = plot_df.copy()
        tbl = tbl[["partner", "value_scaled", "share_pct"]].rename(columns={"partner": "partner", "value_scaled": value_col, "share_pct": "share_of_world_pct"})

        total_val = _scale_trade_value(world_total_th, unit)
        total_row = pd.DataFrame([{"partner": "Total", value_col: total_val, "share_of_world_pct": 100.0 if world_total_th else np.nan}])
        tbl = pd.concat([tbl, total_row], ignore_index=True)

        tbl_disp = tbl.copy()
        for _c in [value_col, "share_of_world_pct"]:
            if _c in tbl_disp.columns:
                tbl_disp[_c] = pd.to_numeric(tbl_disp[_c], errors="coerce").round(2)

        def _bold_total_row(row):
            is_total = str(row.get("partner", "")).strip().lower() == "total"
            return ["font-weight: 800" if is_total else ""] * len(row)

        try:
            sty = tbl_disp.style.apply(_bold_total_row, axis=1).format({value_col: "{:,.2f}", "share_of_world_pct": "{:,.2f}"})
            if hasattr(sty, "hide"):
                sty = sty.hide(axis="index")
            st.dataframe(sty, use_container_width=True)
        except Exception:
            st.dataframe(tbl_disp, use_container_width=True, hide_index=True)

        csv = tbl_disp.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download (Top N + Others) CSV",
            data=csv,
            file_name=f"{title_prefix}_countries_{partner_mode.lower()}_{snap_year}_top{top_n}.csv",
            mime="text/csv",
        )

    # ----- Country trend -----
    with tabs[2]:
        st.markdown(f"## Country trend • {partner_mode} • {snap_year}")
        st.markdown(ctx, unsafe_allow_html=True)

        if partner_mode == "Exports":
            rank_for_opts = exp[exp["year"] == snap_year].copy()
        else:
            rank_for_opts = imp[imp["year"] == snap_year].copy()
        rank_for_opts["value"] = pd.to_numeric(rank_for_opts["value"], errors="coerce")
        rank_for_opts = rank_for_opts[rank_for_opts["partner"].str.lower() != "world"].dropna(subset=["value"]).sort_values("value", ascending=False)

        partner_opts = rank_for_opts["partner"].head(50).tolist()
        if not partner_opts:
            st.info("No partner rows available.")
        else:
            partner = st.selectbox("Select country", partner_opts, index=0, key=f"{key_base}_partner")

            df_src = exp if partner_mode == "Exports" else imp
            df_tr = df_src[df_src["partner"].str.lower() == str(partner).lower()].copy()
            if df_tr.empty:
                st.info("No data available for the selected country.")
            else:
                df_tr = df_tr.groupby("year", as_index=False)["value"].sum()
                df_tr["year"] = pd.to_numeric(df_tr["year"], errors="coerce").astype(int)
                df_tr = df_tr[df_tr["year"].isin(years)].sort_values("year")
                df_tr["value_scaled"] = df_tr["value"].apply(lambda v: _scale_trade_value(v, unit))

                world_series = world_exp if partner_mode == "Exports" else world_imp
                df_tr["world_th"] = [world_series.get(int(y), np.nan) for y in df_tr["year"]]
                df_tr["share_pct"] = np.where(
                    (df_tr["world_th"].notna()) & (df_tr["world_th"] != 0),
                    (df_tr["value"] / df_tr["world_th"]) * 100.0,
                    np.nan,
                )

                y_col = "value_scaled" if metric == "Value" else "share_pct"
                y_title = _trade_unit_label(unit) if metric == "Value" else "Share of World (%)"
                lbl_fmt = "{:,.2f}" if metric == "Value" else "{:,.2f}%"

                fig3 = px.line(df_tr, x="year", y=y_col, markers=True, title=f"Country trend • {partner_mode}: {partner}")
                fig3.update_traces(line=dict(color=(theme.accent if partner_mode == "Exports" else theme.accent2), width=3))
                fig3.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=60, b=10))
                fig3.update_yaxes(title=y_title)
                if show_labels:
                    _add_line_point_labels(fig3, fmt=lbl_fmt)
                st.plotly_chart(fig3, use_container_width=True)

    # ----- Download -----
    with tabs[3]:
        st.markdown("## Download")
        st.markdown(ctx, unsafe_allow_html=True)
        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
        st.markdown("### Download (year range)")

        d1, d2 = st.columns(2)
        with d1:
            dl_start_year = st.selectbox("Start year", years, index=0, key=f"{key_base}_dl_start_year")
        with d2:
            dl_end_year = st.selectbox("End year", years, index=len(years) - 1, key=f"{key_base}_dl_end_year")

        if dl_start_year > dl_end_year:
            dl_start_year, dl_end_year = dl_end_year, dl_start_year

        exp_rng = exp[(exp["year"] >= dl_start_year) & (exp["year"] <= dl_end_year)].copy()
        imp_rng = imp[(imp["year"] >= dl_start_year) & (imp["year"] <= dl_end_year)].copy()

        exp_rng = exp_rng.rename(columns={"value": "exports_thousand"})
        imp_rng = imp_rng.rename(columns={"value": "imports_thousand"})

        dl = exp_rng.merge(imp_rng, on=["partner", "year"], how="outer")
        dl["exports_thousand"] = pd.to_numeric(dl["exports_thousand"], errors="coerce")
        dl["imports_thousand"] = pd.to_numeric(dl["imports_thousand"], errors="coerce")
        dl["balance_thousand"] = dl["exports_thousand"] - dl["imports_thousand"]

        dl = dl.sort_values(["partner", "year"])
        dl["yoy_exports_pct"] = dl.groupby("partner")["exports_thousand"].pct_change() * 100.0
        dl["yoy_imports_pct"] = dl.groupby("partner")["imports_thousand"].pct_change() * 100.0

        dl["exports"] = dl["exports_thousand"].apply(lambda v: _scale_trade_value(v, unit))
        dl["imports"] = dl["imports_thousand"].apply(lambda v: _scale_trade_value(v, unit))
        dl["balance"] = dl["balance_thousand"].apply(lambda v: _scale_trade_value(v, unit))

        out_cols = ["partner", "year", "exports", "imports", "balance", "yoy_exports_pct", "yoy_imports_pct"]
        dl_out = dl[out_cols].copy()

        st.caption(
            f"Download range: {dl_start_year}–{dl_end_year} • Rows: {len(dl_out):,} • "
            f"Unit: {_trade_unit_label(unit)} (base: USD thousand)"
        )
        st.dataframe(dl_out.head(500), use_container_width=True, height=420)

        csv = dl_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download trade (Exports+Imports+Balance + YoY) CSV",
            data=csv,
            file_name=f"{title_prefix}_trade_{dl_start_year}_{dl_end_year}.csv",
            mime="text/csv",
        )


def render_trade_page(
    title: str,
    subtitle: str,
    theme: Theme,
    total_path: str,
    total_hs: str,
    hs6_sheets: List[str],
    unit: str,
    show_labels: bool,
    hero_symbol: str,
) -> None:
    _inject_css(theme)
    _hero(title=title, subtitle=subtitle, theme=theme, right_svg=_svg_coin(hero_symbol, theme))

    # Try to load total series from dedicated Imports/Exports sheets.
    # If not present (some workbooks only contain a single ITC-style HS6 block sheet),
    # fallback to reading the block sheet named exactly like the HS code.
    total_desc = ""
    try:
        imp_total, exp_total = load_trade_total(total_path, total_hs)
    except Exception as e:
        try:
            xls = pd.ExcelFile(total_path)
            if str(total_hs) in [str(s) for s in xls.sheet_names]:
                imp_total, exp_total, total_desc = load_trade_hs6_blocks(total_path, str(total_hs))
            else:
                raise e
        except Exception:
            st.error(f"Could not load trade data for HS {total_hs}: {e}")
            st.caption("Check that the workbook contains identifiable Imports/Exports sheets (or an ITC-style block sheet named as the HS code).")
            return

    hs6_sheets = [str(s) for s in hs6_sheets if str(s) != str(total_hs)]

    # Build tabs: Total + HS6
    tab_labels = ["Total"] + [str(s) for s in hs6_sheets]
    tabs = st.tabs(tab_labels) if len(tab_labels) > 1 else [st.container()]

    # Total tab / container
    with tabs[0]:
        _section(f"Trade — HS {total_hs} (Total)")
        if total_desc:
            short_desc = total_desc.replace("Product:", "").strip()
            if len(short_desc) > 90:
                short_desc = short_desc[:87] + "..."
            st.caption(short_desc)
        render_trade_content(theme, imp_total, exp_total, unit, show_labels, title_prefix=f"hs{total_hs}_total")

    # HS6 tabs
    for i, sh in enumerate(hs6_sheets, start=1):
        if len(tab_labels) > 1:
            ctx = tabs[i]
        else:
            ctx = st.container()
        with ctx:
            try:
                imp_h, exp_h, desc = load_trade_hs6_blocks(total_path, sh)
            except Exception as e:
                st.error(f"Could not read HS6 sheet {sh}: {e}")
                continue
            short_desc = ""
            if desc:
                short_desc = desc.replace("Product:", "").strip()
                if len(short_desc) > 90:
                    short_desc = short_desc[:87] + "..."
            _section(f"Trade — HS {sh}")
            if short_desc:
                st.caption(short_desc)
            render_trade_content(theme, imp_h, exp_h, unit, show_labels, title_prefix=f"hs{sh}")


# -------------------------
# File resolution
# -------------------------
def _resolve_first_existing(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and Path(p).exists():
            return p
    return None

# Resolved data file paths (kept out of sidebar UI)
SILVER_PROD_FILE = _resolve_first_existing([
    "/mnt/data/Silver_Production.xlsx",
    "/mnt/data/Silver Production.xlsx",
    "Silver_Production.xlsx",
    "Silver Production.xlsx",
])

SILVER_TRADE_FILE = _resolve_first_existing([
    "/mnt/data/Silver-7106.xlsx",
    "Silver-7106.xlsx",
])

SILVER_JEW_FILE = _resolve_first_existing([
    "/mnt/data/Silver Jewellery - 711311.xlsx",
    "Silver Jewellery - 711311.xlsx",
])

PLATINUM_FILE = _resolve_first_existing([
    "/mnt/data/Platinum (7110).xlsx",
    "Platinum (7110).xlsx",
])



# -------------------------
# Sidebar
# -------------------------
st.sidebar.markdown("## 🧭 Navigation")
st.sidebar.caption("Silver & Platinum dashboards (theme‑consistent, KPI cards, Top‑N + Others logic).")

NAV_ITEMS = [
    "Silver Production",
    "Silver (HS 7106)",
    "Silver Jewellery (HS 711311)",
    "Platinum (HS 7110)",
]
page = st.sidebar.radio("Go to", NAV_ITEMS, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Display controls")

show_labels = st.sidebar.toggle("Show data labels on charts", value=True, help="Applies to both line and bar charts.")
trade_unit = st.sidebar.selectbox("Trade value unit (base is USD thousand)", options=_trade_unit_options(), index=1)



# -------------------------
# Routing
# -------------------------
if page == "Silver Production":
    theme = THEMES["silver"]
    if not SILVER_PROD_FILE:
        _inject_css(theme)
        _hero("Silver Production Dashboard", "Missing data file: Silver_Production.xlsx", theme, right_svg=_svg_coin("Ag", theme))
        st.error("Silver production file not found. Upload 'Silver_Production.xlsx' (or 'Silver Production.xlsx').")
    else:
        render_silver_production(theme, SILVER_PROD_FILE, show_labels)

elif page == "Silver (HS 7106)":
    theme = THEMES["silver"]
    if not SILVER_TRADE_FILE:
        _inject_css(theme)
        _hero("Silver Trade Dashboard", "Missing data file: Silver-7106.xlsx", theme, right_svg=_svg_coin("Ag", theme))
        st.error("Silver trade file not found. Upload 'Silver-7106.xlsx'.")
    else:
        # HS6 sheets detected in the workbook (only those that are pure digits)
        xls = pd.ExcelFile(SILVER_TRADE_FILE)
        hs6 = [s for s in xls.sheet_names if re.fullmatch(r"\d{6}", str(s))]
        render_trade_page(
            title="Silver Trade Dashboard",
            subtitle="HS 7106 — Silver (incl. silver plated with gold or platinum)",
            theme=theme,
            total_path=SILVER_TRADE_FILE,
            total_hs="7106",
            hs6_sheets=hs6,
            unit=trade_unit,
            show_labels=show_labels,
            hero_symbol="Ag",
        )

elif page == "Silver Jewellery (HS 711311)":
    theme = THEMES["silver"]
    if not SILVER_JEW_FILE:
        _inject_css(theme)
        _hero("Silver Jewellery Trade", "Missing data file: Silver Jewellery - 711311.xlsx", theme, right_svg=_svg_coin("Ag", theme))
        st.error("Silver jewellery file not found. Upload 'Silver Jewellery - 711311.xlsx'.")
    else:
        # The HS sheet is named "711311"
        render_trade_page(
            title="Silver Jewellery Trade Dashboard",
            subtitle="HS 711311 — Articles of jewellery and parts, of silver (latest ITC-style series)",
            theme=theme,
            total_path=SILVER_JEW_FILE,
            total_hs="711311",
            hs6_sheets=[],  # already HS6
            unit=trade_unit,
            show_labels=show_labels,
            hero_symbol="Ag",
        )

elif page == "Platinum (HS 7110)":
    theme = THEMES["platinum"]
    if not PLATINUM_FILE:
        _inject_css(theme)
        _hero("Platinum Trade Dashboard", "Missing data file: Platinum (7110).xlsx", theme, right_svg=_svg_coin("Pt", theme))
        st.error("Platinum trade file not found. Upload 'Platinum (7110).xlsx'.")
    else:
        xls = pd.ExcelFile(PLATINUM_FILE)
        hs6 = [s for s in xls.sheet_names if re.fullmatch(r"\d{6}", str(s))]
        render_trade_page(
            title="Platinum Trade Dashboard",
            subtitle="HS 7110 — Platinum (unwrought/powder/semi‑manufactured etc.)",
            theme=theme,
            total_path=PLATINUM_FILE,
            total_hs="7110",
            hs6_sheets=hs6,
            unit=trade_unit,
            show_labels=show_labels,
            hero_symbol="Pt",
        )
