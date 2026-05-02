"""
Dashboard: avaliações físicas TCE (Excel) com PCA e radares por atleta e por escalão.

Ficheiro esperado: mesma estrutura que "Avaliações Físicas TCE 2025-2026.xlsx"
(folha "Avaliações Físicas", linha 1 = cabeçalhos dos testes).

Defina o caminho via variável de ambiente TCE_XLSX ou edite DEFAULT_XLSX abaixo.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Caminho por omissão: primeiro tenta na pasta data do projeto, depois Downloads.
_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_DOWNLOADS = Path.home() / "Downloads" / "Avaliações Físicas TCE 2025-2026.xlsx"
_DEFAULT_MEIO = _REPO / "data" / "Avaliações Físicas TCE_2025-2026_Inicio_e_Meio_epoca.xlsx"
_DEFAULT_BASE = _REPO / "data" / "Avaliações Físicas TCE 2025-2026.xlsx"
if _DEFAULT_MEIO.is_file():
    DEFAULT_XLSX = _DEFAULT_MEIO
elif _DEFAULT_BASE.is_file():
    DEFAULT_XLSX = _DEFAULT_BASE
elif _DEFAULT_DOWNLOADS.is_file():
    DEFAULT_XLSX = _DEFAULT_DOWNLOADS
else:
    DEFAULT_XLSX = _DEFAULT_MEIO

# Aumentar quando mudarem colunas de scores / PCA (força o Streamlit a ignorar cache antigo).
PCA_SCHEMA_VERSION = 5


def imputar_numericas_por_escalão(
    df: pd.DataFrame,
    colunas: list[str],
    grupo_col: str = "Escalão",
) -> pd.DataFrame:
    """
    Preenche ausentes com a média do mesmo escalão; o que ficar em falta usa média global da coluna.
    Coluna totalmente vazia → 0.0 (evita erros no PCA).
    """
    out = df.copy()
    g = out[grupo_col]

    for c in colunas:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        filled = s.astype("float64").copy()
        for label in g.dropna().unique():
            mask = g == label
            m_esc = s[mask].mean()
            if pd.notna(m_esc):
                na_in_group = mask & filled.isna()
                filled.loc[na_in_group] = m_esc
        if filled.isna().any():
            m_glob = s.mean()
            if pd.notna(m_glob):
                filled = filled.fillna(m_glob)
            else:
                filled = filled.fillna(0.0)
        out[c] = filled
    return out


def parse_tempo_segundos(x) -> float | None:
    """Converte tempos tipo 11''62 ou 3,86 para segundos (float)."""
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    if "''" in s:
        s = s.replace("''", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_reacao_ms_pair(cell) -> tuple[float | None, float | None]:
    """Ex.: '333 ms (E) / 500 ms (D)' -> (333.0, 500.0)."""
    if pd.isna(cell):
        return None, None
    parts = str(cell).split("/")
    if len(parts) != 2:
        return None, None

    def ms_token(tok: str) -> float | None:
        t = tok.strip()
        if "ms" not in t.lower():
            return None
        num = "".join(ch for ch in t.split("ms")[0] if ch.isdigit())
        return float(num) if num else None

    return ms_token(parts[0]), ms_token(parts[1])


def parse_grau_espargata(x) -> int | None:
    if pd.isna(x):
        return None
    s = str(x).strip().split("/")[0].split("º")[0].strip()
    try:
        return int(s)
    except ValueError:
        return None


def parse_grau_slash(x, idx: int) -> int | None:
    if pd.isna(x):
        return None
    parts = str(x).split("/")
    if len(parts) <= idx:
        return None
    s = parts[idx].strip().split("º")[0].strip()
    try:
        return int(s)
    except ValueError:
        return None


def calcular_fatores(
    data: pd.DataFrame,
    colunas: list[str],
    n_fatores: int = 1,
    invert_cols: list[str] | None = None,
    groups: pd.Series | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    PCA + MinMax [0,1] por grupo (`groups`) quando fornecido; caso contrário escala global.
    """
    imputer = SimpleImputer(strategy="mean")
    X_df = data[colunas].copy()

    if invert_cols:
        for c in invert_cols:
            if c in X_df.columns:
                X_df[c] = X_df[c].apply(lambda v: -v if pd.notnull(v) else v)

    X = imputer.fit_transform(X_df)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_fatores)
    scores = pca.fit_transform(X_scaled)

    col_names = [f"Fator{i+1}" for i in range(n_fatores)]
    scores_df = pd.DataFrame(scores, columns=col_names)

    if groups is not None:
        g = groups.fillna("__MISSING__").astype(str).values
        scaled_scores = np.full(scores.shape, np.nan, dtype=float)
        for i, comp in enumerate(col_names):
            for label in np.unique(g):
                mask = g == label
                vals = scores_df.loc[mask, comp].values.reshape(-1, 1)
                if vals.size == 0:
                    continue
                if vals.size == 1 or np.nanstd(vals) < 1e-12:
                    continue
                try:
                    scaled_scores[mask, i] = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                        vals
                    ).ravel()
                except Exception:
                    pass
            col = scaled_scores[:, i]
            missing = np.isnan(col)
            if missing.any():
                fallback = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                    scores_df.iloc[:, i].values.reshape(-1, 1)
                ).ravel()
                col[missing] = fallback[missing]
                scaled_scores[:, i] = col
        scores = scaled_scores
    else:
        scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores)

    loadings = pd.DataFrame(
        pca.components_.T, index=colunas, columns=col_names
    )
    return scores, loadings


def load_and_prepare(xlsx_path: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name="Avaliações Físicas", header=1)
    rename = {
        "Unnamed: 0": "Atleta",
        "Unnamed: 1": "Sexo",
        "Unnamed: 2": "Idade",
        "Unnamed: 3": "Categoria",
        "Unnamed: 4": "Fase",
        "Unnamed: 5": "Data",
    }
    data = raw.rename(columns=rename)
    drop_tail = [c for c in raw.columns if str(c).startswith("Unnamed:") and c not in rename]
    data = data.drop(columns=[c for c in drop_tail if c in data.columns], errors="ignore")
    data = data.dropna(subset=["Atleta"])
    data = data[data["Atleta"].astype(str).str.strip() != "Unnamed: 0"]

    data["Escalão"] = (
        data["Categoria"].astype(str).str.strip()
        + "_"
        + data["Sexo"].astype(str).str.strip()
    )

    data["tempo_4x10m_s"] = data["4x10m"].map(parse_tempo_segundos)
    data["tempo_20m_s"] = data["20m"].map(parse_tempo_segundos)
    r_esq, r_dir = zip(*data["Reação + pontapé (s)"].map(parse_reacao_ms_pair))
    data["reacao_pontape_esq"] = r_esq
    data["reacao_pontape_dir"] = r_dir

    for col in ["Impulsão horizontal", "Impulsão vertical", "Abdominais máx", "Flexões máx"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Sit-and-Reach (cm)"] = pd.to_numeric(
        data["Sit-and-Reach (cm)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    data["espargata_frontal"] = data["Espargata frontal (°)"].map(parse_grau_espargata)
    data["espargata_lat_dir"] = data["Espargata lateral D/E (°)"].map(
        lambda x: parse_grau_slash(x, 0)
    )
    data["espargata_lat_esq"] = data["Espargata lateral D/E (°)"].map(
        lambda x: parse_grau_slash(x, 1)
    )
    data["amplitude_pontape_dir"] = data["Amplitude pontapés D/E (°)"].map(
        lambda x: parse_grau_slash(x, 0)
    )
    data["amplitude_pontape_esq"] = data["Amplitude pontapés D/E (°)"].map(
        lambda x: parse_grau_slash(x, 1)
    )

    data["Vaivém"] = pd.to_numeric(data["Vaivém"], errors="coerce")
    data["Pontapés 1 min"] = pd.to_numeric(data["Pontapés 1 min"], errors="coerce")

    return data


@st.cache_data
def load_and_compute_scores(xlsx_path: str, schema_version: int) -> pd.DataFrame:
    """Carrega o Excel e devolve o DataFrame com fatores PCA (cache Streamlit)."""
    _ = schema_version  # entra na chave de cache; aumentar PCA_SCHEMA_VERSION invalida entradas antigas
    return compute_factor_scores(load_and_prepare(xlsx_path))


def compute_factor_scores(data: pd.DataFrame) -> pd.DataFrame:
    groups = data["Escalão"]

    resistencia_cols = ["Vaivém", "Pontapés 1 min"]
    velocidade_cols = [
        "tempo_4x10m_s",
        "tempo_20m_s",
        "reacao_pontape_esq",
        "reacao_pontape_dir",
    ]
    forca_cols = [
        "Impulsão horizontal",
        "Impulsão vertical",
        "Abdominais máx",
        "Flexões máx",
    ]
    flex_cols = [
        "Sit-and-Reach (cm)",
        "espargata_frontal",
        "espargata_lat_dir",
        "espargata_lat_esq",
        "amplitude_pontape_dir",
        "amplitude_pontape_esq",
    ]

    pca_input_cols = list(
        dict.fromkeys(resistencia_cols + velocidade_cols + forca_cols + flex_cols)
    )
    data_imp = imputar_numericas_por_escalão(data, pca_input_cols, grupo_col="Escalão")

    resistencia_score, _ = calcular_fatores(
        data_imp, resistencia_cols, n_fatores=1, groups=groups
    )
    velocidade_score, _ = calcular_fatores(
        data_imp,
        velocidade_cols,
        n_fatores=1,
        invert_cols=velocidade_cols,
        groups=groups,
    )
    forca_score, _ = calcular_fatores(data_imp, forca_cols, n_fatores=1, groups=groups)
    flex_score, _ = calcular_fatores(data_imp, flex_cols, n_fatores=1, groups=groups)

    # Mesmo PCA com MinMax global (todos os escalões) — para comparar escalões no mesmo radar.
    resistencia_g, _ = calcular_fatores(data_imp, resistencia_cols, n_fatores=1, groups=None)
    velocidade_g, _ = calcular_fatores(
        data_imp,
        velocidade_cols,
        n_fatores=1,
        invert_cols=velocidade_cols,
        groups=None,
    )
    forca_g, _ = calcular_fatores(data_imp, forca_cols, n_fatores=1, groups=None)
    flex_g, _ = calcular_fatores(data_imp, flex_cols, n_fatores=1, groups=None)

    out = data_imp.copy()
    out["F_resistencia"] = resistencia_score[:, 0]
    out["F_velocidade"] = velocidade_score[:, 0]
    out["F_forca"] = forca_score[:, 0]
    out["F_flex"] = flex_score[:, 0]
    out["Fg_resistencia"] = resistencia_g[:, 0]
    out["Fg_velocidade"] = velocidade_g[:, 0]
    out["Fg_forca"] = forca_g[:, 0]
    out["Fg_flex"] = flex_g[:, 0]
    return out


def radar_figure(
    valores: list[float],
    title: str,
    name: str,
) -> go.Figure:
    labels = [
        "Resistência",
        "Velocidade",
        "Força",
        "Flexibilidade",
    ]
    vals = list(valores) + [valores[0]]
    thetas = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=vals, theta=thetas, fill="toself", name=name)
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=False, range=[0, 1])
        ),
        title=title,
        showlegend=True,
        margin=dict(l=48, r=120, t=64, b=48),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
        ),
    )
    return fig


def radar_figure_compare(
    valores_a: list[float],
    valores_b: list[float],
    name_a: str,
    name_b: str,
    title: str,
) -> go.Figure:
    labels = ["Resistência", "Velocidade", "Força", "Flexibilidade"]
    thetas = labels + [labels[0]]
    va = list(valores_a) + [valores_a[0]]
    vb = list(valores_b) + [valores_b[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=va,
            theta=thetas,
            fill="toself",
            name=name_a,
            fillcolor="rgba(30, 136, 229, 0.35)",
            line=dict(color="rgb(30, 136, 229)"),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=vb,
            theta=thetas,
            fill="toself",
            name=name_b,
            fillcolor="rgba(230, 120, 0, 0.35)",
            line=dict(color="rgb(200, 90, 0)"),
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=False, range=[0, 1])
        ),
        title=title,
        margin=dict(l=48, r=120, t=64, b=48),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
        ),
    )
    return fig


def _fatores_row(row: pd.Series) -> list[float]:
    return [
        float(row["F_resistencia"]),
        float(row["F_velocidade"]),
        float(row["F_forca"]),
        float(row["F_flex"]),
    ]


def _norm_fase(s) -> str:
    return str(s).strip().lower()


def _titulo_momento_epoca(fase_val: str) -> str:
    """Rótulo legível para a coluna Fase (altura / momento da época)."""
    k = _norm_fase(fase_val)
    if k == "inicio":
        return "Início da época"
    if k == "meio":
        return "Meio da época"
    t = str(fase_val).strip()
    return t if t else "—"


def _fases_ordenadas(series: pd.Series) -> list[str]:
    vals = series.dropna().astype(str).unique().tolist()

    def sort_key(f: str) -> tuple[int, str]:
        k = _norm_fase(f)
        order = {"inicio": 0, "meio": 1}
        return (order.get(k, 50), k)

    return sorted(vals, key=sort_key)


def main():
    st.set_page_config(page_title="Avaliações TCE — PCA", layout="wide")
    st.title("Avaliações físicas TCE — PCA e radares")
    st.sidebar.caption(
        "Ficheiro com **Início + Meio época**: `data/Avaliações Físicas TCE_2025-2026_Inicio_e_Meio_epoca.xlsx`. "
    )

    env_path = os.environ.get("TCE_XLSX", "")
    default = env_path if env_path else str(DEFAULT_XLSX)
    xlsx_path = st.sidebar.text_input("Caminho do Excel (.xlsx)", value=default)

    if not xlsx_path or not Path(xlsx_path).is_file():
        st.error(
            "Ficheiro não encontrado. Coloque o Excel em `data/` ou aponte o caminho completo "
            "(ou defina a variável de ambiente TCE_XLSX)."
        )
        return

    df = load_and_compute_scores(xlsx_path, PCA_SCHEMA_VERSION)
    if "F_forca" not in df.columns or "Fg_forca" not in df.columns:
        # Sessões com cache corrompido / muito antigo
        df = compute_factor_scores(load_and_prepare(xlsx_path))

    tab_atleta, tab_escalao, tab_raw = st.tabs(
        ["Radar por atleta", "Radar por escalão", "Dados / colunas PCA"]
    )

    with tab_atleta:
        nomes = sorted(df["Atleta"].astype(str).unique().tolist())
        escolha = st.selectbox("Atleta", options=nomes, index=0)
        sub_df = df[df["Atleta"].astype(str) == escolha]

        def _row_fase(fase_alvo: str) -> pd.Series | None:
            m = sub_df["Fase"].astype(str).str.strip().str.lower() == fase_alvo
            if not m.any():
                return None
            return sub_df.loc[m].iloc[0]

        r_inicio = _row_fase("inicio")
        r_meio = _row_fase("meio")

        if r_inicio is not None and r_meio is not None:
            fig = radar_figure_compare(
                _fatores_row(r_inicio),
                _fatores_row(r_meio),
                "Início",
                "Meio época",
                f"{escolha} — comparação de fases ({r_inicio['Escalão']})",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            if r_inicio is not None:
                row = r_inicio
            elif r_meio is not None:
                row = r_meio
            else:
                row = sub_df.iloc[0]
            vals = _fatores_row(row)
            sub = f"{row['Escalão']} · {row.get('Fase', '')}"
            fig = radar_figure(vals, title=f"{row['Atleta']} ({sub})", name=row["Atleta"])
            st.plotly_chart(fig, use_container_width=True)
            if r_inicio is None or r_meio is None:
                st.info(
                    "Só há uma fase para este atleta no ficheiro. "
                    "Usa o Excel `..._Inicio_e_Meio_epoca.xlsx` para ver Início vs Meio no radar."
                )

    with tab_escalao:
        escaloes = sorted(df["Escalão"].dropna().unique().tolist())
        fases_disp = _fases_ordenadas(df["Fase"])
        modo = st.radio(
            "Visualização",
            [
                "Comparar escalões (médias)",
                "Início vs Meio (mesmo escalão)",
            ],
            horizontal=True,
        )
        st.caption(
            "As comparações por escalão usam o **momento da época** (coluna *Fase*: "
            "Início / Meio). Há **um radar por momento**, para não misturar avaliações."
        )

        cols_f = ["F_resistencia", "F_velocidade", "F_forca", "F_flex"]
        cols_fg = ["Fg_resistencia", "Fg_velocidade", "Fg_forca", "Fg_flex"]
        labels = ["Resistência", "Velocidade", "Força", "Flexibilidade"]
        thetas = labels + [labels[0]]

        if modo == "Início vs Meio (mesmo escalão)":
            esc = st.selectbox("Escalão (Categoria_Sexo)", options=escaloes, key="esc_inme")
            g_in = df[
                (df["Escalão"] == esc)
                & (df["Fase"].astype(str).str.strip().str.lower() == "inicio")
            ]
            g_me = df[
                (df["Escalão"] == esc)
                & (df["Fase"].astype(str).str.strip().str.lower() == "meio")
            ]
            if g_in.empty or g_me.empty:
                st.warning(
                    "Para este escalão é preciso haver avaliações **Início** e **Meio** no Excel."
                )
            else:
                mi = g_in[cols_f].mean()
                mm = g_me[cols_f].mean()
                fig = radar_figure_compare(
                    [float(mi[c]) for c in cols_f],
                    [float(mm[c]) for c in cols_f],
                    f"Início (n={len(g_in)})",
                    f"Meio época (n={len(g_me)})",
                    f"Média por fase — {esc}",
                )
                st.plotly_chart(fig, use_container_width=True)
        elif modo == "Comparar escalões (médias)":
            pick = st.multiselect(
                "Escalões a comparar (em cada momento da época)",
                options=escaloes,
                default=escaloes[: min(3, len(escaloes))],
                key="esc_multi_compare",
            )
            if not fases_disp:
                st.warning("Não há valores de **Fase** nos dados.")
            elif not pick:
                st.info("Seleciona pelo menos um escalão.")
            else:
                st.caption(
                    "Neste modo os radares usam scores **Fg_*** (MinMax **global** em todas as "
                    "linhas), para os escalões serem comparáveis entre si. Os scores **F_*** "
                    "(por escalão) mantêm-se nos outros separadores."
                )

                def _fig_comparar_escaloes_por_fase(fase_raw: str) -> go.Figure | None:
                    dff = df[df["Fase"].astype(str).map(_norm_fase) == _norm_fase(fase_raw)]
                    fig = go.Figure()
                    for esc in pick:
                        sub = dff[dff["Escalão"] == esc]
                        if sub.empty:
                            continue
                        m = sub[cols_fg].mean()
                        vals = [
                            float(m["Fg_resistencia"]),
                            float(m["Fg_velocidade"]),
                            float(m["Fg_forca"]),
                            float(m["Fg_flex"]),
                        ]
                        vals = vals + [vals[0]]
                        fig.add_trace(
                            go.Scatterpolar(
                                r=vals,
                                theta=thetas,
                                fill="toself",
                                name=f"{esc} (n={len(sub)})",
                            )
                        )
                    if not fig.data:
                        return None
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True, showticklabels=False, range=[0, 1]
                            )
                        ),
                        title=f"Comparação de escalões — {_titulo_momento_epoca(fase_raw)}",
                        margin=dict(l=48, r=140, t=64, b=48),
                        legend=dict(
                            orientation="v",
                            yanchor="middle",
                            y=0.5,
                            xanchor="left",
                            x=1.02,
                            font=dict(size=11),
                        ),
                    )
                    return fig

                for i in range(0, len(fases_disp), 2):
                    left, right = st.columns(2)
                    with left:
                        fr = fases_disp[i]
                        st.subheader(_titulo_momento_epoca(fr))
                        fig_l = _fig_comparar_escaloes_por_fase(fr)
                        if fig_l is None:
                            st.caption(
                                "Nenhum dos escalões escolhidos tem dados neste momento da época."
                            )
                        else:
                            st.plotly_chart(fig_l, use_container_width=True)
                    with right:
                        if i + 1 < len(fases_disp):
                            fr2 = fases_disp[i + 1]
                            st.subheader(_titulo_momento_epoca(fr2))
                            fig_r = _fig_comparar_escaloes_por_fase(fr2)
                            if fig_r is None:
                                st.caption(
                                    "Nenhum dos escalões escolhidos tem dados neste momento da época."
                                )
                            else:
                                st.plotly_chart(fig_r, use_container_width=True)

    with tab_raw:
        st.subheader("Colunas usadas nas dimensões PCA (TCE)")
        st.markdown(
            """
- **Resistência:** Vaivém, Pontapés 1 min  
- **Velocidade:** 4x10m (s), 20 m (s), reação pontapé E/D (ms) — valores invertidos antes do PCA (menor tempo/reação = melhor)  
- **Força (1 fator):** Impulsão horizontal, Impulsão vertical, Abdominais máx, Flexões máx  
- **Flexibilidade:** Sit-and-Reach, espargatas, amplitudes de pontapé  

Valores em falta nas variáveis do PCA são imputados com a **média do escalão** e, se ainda faltar, com a **média global** da coluna (coluna toda vazia → 0). O PCA continua a usar `SimpleImputer` como rede de segurança.

- **F_*** (MinMax **por escalão**): comparação justa **dentro** do mesmo Categoria+Sexo (ex.: atleta vs atleta, Início vs Meio no mesmo escalão). O PCA usa sempre todas as linhas.
- **Fg_*** (MinMax **global**): mesmos fatores PCA, mas o 0–1 é calculado **em todo o conjunto** — adequado para **comparar escalões diferentes** no modo «Comparar escalões».

No separador **Radar por escalão**, a comparação entre escalões usa **Fg_*** e é feita **por momento da época** (*Fase*): um gráfico para **Início da época** e outro para **Meio da época** (e mais, se existirem outras fases nos dados).
            """
        )
        show = [
            "Atleta",
            "Fase",
            "Sexo",
            "Categoria",
            "Escalão",
            "F_resistencia",
            "F_velocidade",
            "F_forca",
            "F_flex",
            "Fg_resistencia",
            "Fg_velocidade",
            "Fg_forca",
            "Fg_flex",
        ]
        st.dataframe(
            df[show].sort_values(["Atleta", "Fase"]),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
