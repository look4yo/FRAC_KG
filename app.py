import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
import tempfile
import os
import pandas as pd
import re
import math


st.set_page_config(layout="wide", page_title="Neo4j KG Viewer")

# -----------------------------
# 连接 Neo4j（建议用 st.secrets 管理）
# -----------------------------
NEO4J_URI = st.secrets.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = st.secrets.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD", "10129926")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

st.title("Knowledge Graph Viewer")

# -----------------------------
# 内置映射文件：来自 Polarity_mapping_table.xlsx
# -----------------------------
MAPPING_FILENAME = "Polarity_mapping_table.xlsx"
MAPPING_SHEET = "Polarity_Mapping_96"

# -----------------------------
# 统计遍历关系类型：固定内置（不在 UI 展示）
# -----------------------------
TRAVERSAL_TYPES_FOR_STATS = ["Increases", "Decreases", "Causes"]


# -----------------------------
# Center node selector (two-level dropdown)
#   Level-1: label in {Component, Mechanism, Mix, Property}
#   Level-2: node names under the selected label
# Note:
#   - This is ONLY for UI selection; downstream queries still use name-based matching as before.
# -----------------------------
CENTER_LABEL_OPTIONS = ["Component", "Mechanism", "Mix", "Property"]
_CENTER_LABEL_SET = set(CENTER_LABEL_OPTIONS)

def _safe_label(label: str) -> str:
    lab = (label or "").strip()
    return lab if lab in _CENTER_LABEL_SET else "Component"

@st.cache_data(show_spinner=False)
def fetch_node_names_by_label(label: str) -> list:
    lab = _safe_label(label)
    cypher = f"""
    MATCH (n:`{lab}`)
    WHERE n.name IS NOT NULL AND trim(toString(n.name)) <> ''
    RETURN DISTINCT toString(n.name) AS name
    ORDER BY name
    """
    with driver.session() as session:
        rows = session.run(cypher).data()
    return [r["name"] for r in rows if r.get("name")]


# -----------------------------
# Session state init
# -----------------------------
if "nodes" not in st.session_state:
    st.session_state.nodes = []
if "edges" not in st.session_state:
    st.session_state.edges = []
if "rel_types" not in st.session_state:
    st.session_state.rel_types = []
if "rel_selected" not in st.session_state:
    st.session_state.rel_selected = []  # 空 => 视为不过滤（显示全部）

# Global scoring cache
if "global_stats" not in st.session_state:
    st.session_state.global_stats = None
if "global_detail_df" not in st.session_state:
    st.session_state.global_detail_df = None

# Path-level scoring cache (3.3.3)
if "path_scores_df" not in st.session_state:
    st.session_state.path_scores_df = None
if "path_raw_counts_df" not in st.session_state:
    st.session_state.path_raw_counts_df = None



# -----------------------------
# 取 k-hop 子图数据（用于可视化）
# -----------------------------
def fetch_subgraph(center_name: str, k_hop: int, max_paths: int):
    k_hop = int(k_hop)
    if k_hop < 1:
        k_hop = 1
    if k_hop > 5:
        k_hop = 5

    cypher = f"""
    MATCH (n {{name: $center}})
    MATCH p=(n)-[*1..{k_hop}]-(m)
    RETURN p
    LIMIT $limit
    """

    nodes = {}
    edges = {}

    with driver.session() as session:
        records = session.run(cypher, center=center_name, limit=int(max_paths))
        for r in records:
            p = r["p"]
            for n in p.nodes:
                nid = str(n.id)
                if nid not in nodes:
                    nodes[nid] = {
                        "id": nid,
                        "labels": list(n.labels),
                        "props": dict(n)
                    }
            for rel in p.relationships:
                rid = str(rel.id)
                if rid not in edges:
                    edges[rid] = {
                        "id": rid,
                        "source": str(rel.start_node.id),
                        "target": str(rel.end_node.id),
                        "type": rel.type,
                        "props": dict(rel)
                    }

    return list(nodes.values()), list(edges.values())


def build_pyvis(nodes, edges, center_name, rel_allow):
    """
    rel_allow 为空列表时 => 不过滤（显示全部边）
    rel_allow 非空时   => 只显示在 rel_allow 中的边
    """
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

    for n in nodes:
        name = n["props"].get("name", n["id"])
        labels = n["labels"]

        size = 12
        color = "#f4b942"  # 默认黄色

        if name == center_name:
            size = 30
            color = "#4a86ff"  # 蓝色（中心）

        title = f"labels: {labels}\nprops: {n['props']}"
        net.add_node(n["id"], label=name, title=title, color=color, size=size)

    for e in edges:
        if rel_allow and (e["type"] not in rel_allow):
            continue
        net.add_edge(
            e["source"], e["target"],
            label=e["type"],
            title=str(e["props"]),
            color="#bdbdbd",
            arrows="to"
        )

    return net


# -----------------------------
# 3.3.1 全局统计：取 “center -> (k hop) -> Property” 证据
# - 只沿 traversalTypes 走（内置 Increases/Decreases/Causes）
# - 取路径最后一条关系类型 lastType，用于 polarity 映射
# -----------------------------
def fetch_property_lasttype_evidence(center_name: str, k_hop: int, max_paths: int, traversal_types: list[str]):
    """
    返回 DataFrame: columns = [property, lastType]
    注意：Neo4j 不允许在可变长度上界使用参数，因此需要把 k_hop 写死进 cypher 字符串。
    """
    k_hop = int(k_hop)
    if k_hop < 1:
        k_hop = 1
    if k_hop > 7:
        k_hop = 7

    cypher = f"""
    MATCH (c {{name:$center}})
    MATCH p=(c)-[r*1..{k_hop}]->(prop:Property)
    WHERE all(x IN r WHERE type(x) IN $trav)
    WITH prop, last(r) AS lr
    RETURN prop.name AS property, type(lr) AS lastType
    LIMIT $limit
    """

    with driver.session() as session:
        rows = session.run(
            cypher,
            center=center_name,
            trav=traversal_types,
            limit=int(max_paths),
        ).data()

    if not rows:
        return pd.DataFrame(columns=["property", "lastType"])
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_preference_map_from_excel():
    """
    读取 Polarity_mapping_table.xlsx / Polarity_Mapping_96
    关键列：
      - PropertyName
      - PolarityType: Higher-is-better / Lower-is-better / Optimal/Uncertain

    映射到 preference:
      - Higher-is-better -> higher
      - Lower-is-better  -> lower
      - Optimal/Uncertain-> range
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(base_dir, MAPPING_FILENAME)

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"Cannot find {MAPPING_FILENAME} in: {base_dir}\n"
            f"Please place {MAPPING_FILENAME} next to apptest.py."
        )

    df = pd.read_excel(xlsx_path, sheet_name=MAPPING_SHEET)

    cols = {c.strip().lower(): c for c in df.columns}
    prop_col = cols.get("propertyname")
    pol_col = cols.get("polaritytype")
    dir_col = cols.get("beneficialdirection")  # 可选，用于展示

    if not prop_col or not pol_col:
        raise ValueError(
            f"{MAPPING_FILENAME}/{MAPPING_SHEET} must contain columns: "
            f"'PropertyName' and 'PolarityType'."
        )

    def pol_to_pref(pol: str) -> str:
        s = str(pol).strip().lower()
        if "higher" in s:
            return "higher"
        if "lower" in s:
            return "lower"
        return "range"

    mp = {}
    for _, row in df.iterrows():
        prop = str(row[prop_col]).strip()
        pol = row[pol_col]
        if not prop:
            continue
        mp[prop] = pol_to_pref(pol)

    show_cols = [prop_col, pol_col]
    if dir_col:
        show_cols.append(dir_col)
    mapping_df = df[show_cols].rename(
        columns={prop_col: "PropertyName", pol_col: "PolarityType", (dir_col or ""): "BeneficialDirection"}
    )

    return mp, mapping_df


def map_evidence_to_posnegunc(last_type: str, preference: str):
    """
    desirability-aware mapping:
      - higher: Increases->pos, Decreases->neg
      - lower : Decreases->pos, Increases->neg
      - range : always unc
      - non (Increases/Decreases): unc
    """
    lt = (last_type or "").strip()
    pref = (preference or "").strip().lower()

    if pref == "range":
        return "unc"

    if lt not in ("Increases", "Decreases"):
        return "unc"

    if pref == "lower":
        return "pos" if lt == "Decreases" else "neg"
    return "pos" if lt == "Increases" else "neg"


def compute_global_property_level_stats(evidence_df: pd.DataFrame, pref_map: dict):
    """
    property-level 汇总：
      - 同一 property 若同时出现 pos 与 neg => final_class = unc（冲突）
      - 只有 pos => pos；只有 neg => neg；其他 => unc

    netscore = (Npos - Nneg) / (Npos + Nneg + 2)
    """
    if evidence_df.empty:
        return (
            {"Nprop": 0, "Npos": 0, "Nneg": 0, "Nunc": 0, "netscore": 0.0},
            pd.DataFrame(columns=["property", "preference", "pos_cnt", "neg_cnt", "unc_cnt", "final_class"])
        )

    tmp = evidence_df.copy()
    tmp["preference"] = tmp["property"].map(pref_map).fillna("range")
    tmp["polarity"] = tmp.apply(lambda r: map_evidence_to_posnegunc(r["lastType"], r["preference"]), axis=1)

    pivot = (
        tmp.groupby(["property", "preference"])["polarity"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("pos", "neg", "unc"):
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot.rename(columns={"pos": "pos_cnt", "neg": "neg_cnt", "unc": "unc_cnt"})

    def final_class(row):
        has_pos = row["pos_cnt"] > 0
        has_neg = row["neg_cnt"] > 0
        if has_pos and has_neg:
            return "unc"
        if has_pos:
            return "pos"
        if has_neg:
            return "neg"
        return "unc"

    pivot["final_class"] = pivot.apply(final_class, axis=1)

    Nprop = int(pivot.shape[0])
    Npos = int((pivot["final_class"] == "pos").sum())
    Nneg = int((pivot["final_class"] == "neg").sum())
    Nunc = int((pivot["final_class"] == "unc").sum())

    denom = (Npos + Nneg + 2)
    netscore = float((Npos - Nneg) / denom) if denom != 0 else 0.0

    stats = {"Nprop": Nprop, "Npos": Npos, "Nneg": Nneg, "Nunc": Nunc, "netscore": round(netscore, 4)}
    detail_df = pivot[["property", "preference", "pos_cnt", "neg_cnt", "unc_cnt", "final_class"]].sort_values(
        by=["final_class", "property"], ascending=[True, True]
    )

    return stats, detail_df



# -----------------------------
# 3.3.3 路径级分析：Eqs. (17)-(20) 复现（Top-10 fibers）
# 约束：
#  - 中间节点仅允许 :Mechanism
#  - Causes 仅桥接（即末端必须是 Increases/Decreases）
#  - 去重口径：DISTINCT by (node_ids + rel_types)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_path_counts_for_targets(k_hop: int, max_paths: int, targets: list[str], traversal_types: list[str]):
    """
    返回 DataFrame:
      columns = [fiber, target, hop, lastType, nPaths]

    仅返回满足以下条件的路径：
      - path = (fiber)-[*1..k]->(target:Property)
      - all rel.type in traversal_types
      - last rel.type in {Increases, Decreases}
      - all intermediate nodes (excluding endpoints) have :Mechanism label
      - DISTINCT by (node_ids + rel_types)

    说明：
      - 这里把“fiber”定义为 (f:Component) 且 name 包含 'fiber'（大小写不敏感）。
        若你后续希望更严格的 fiber 节点集合，可以把 where 条件改成 (f:Fiber)。
    """
    k_hop = int(k_hop)
    if k_hop < 1:
        k_hop = 1
    if k_hop > 3:
        k_hop = 3

    targets = [str(t).strip() for t in (targets or []) if str(t).strip()]
    if not targets:
        return pd.DataFrame(columns=["fiber", "target", "hop", "lastType", "nPaths"])

    cypher = f"""
    MATCH (f:Component)
    WHERE toLower(f.name) CONTAINS 'fiber'
    MATCH path=(f)-[r*1..{k_hop}]->(p:Property)
    WHERE p.name IN $targets
      AND all(x IN r WHERE type(x) IN $trav)
      AND type(last(r)) IN ['Increases','Decreases']
      AND all(n IN nodes(path)[1..-1] WHERE n:Mechanism)
    WITH f.name AS fiber, p.name AS target,
         length(path) AS hop,
         type(last(r)) AS lastType,
         [n IN nodes(path) | id(n)] AS node_ids,
         [x IN r | type(x)] AS rel_types
    WITH DISTINCT fiber, target, hop, lastType, node_ids, rel_types
    WITH fiber, target, hop, lastType, count(*) AS nPaths
    RETURN fiber, target, hop, lastType, nPaths
    ORDER BY fiber, target, hop
    LIMIT $limit
    """

    with driver.session() as session:
        rows = session.run(
            cypher,
            targets=targets,
            trav=traversal_types,
            limit=int(max_paths),
        ).data()

    if not rows:
        return pd.DataFrame(columns=["fiber", "target", "hop", "lastType", "nPaths"])
    return pd.DataFrame(rows)


def compute_path_level_scores(df_counts: pd.DataFrame, pref_map: dict, target1: str, target2: str, alpha: float = 1.0):
    """
    根据论文 Eqs. (17)-(20) 计算 Top-10 fibers 的综合评分。

    需要的计数字段（由 df_counts 提供）：
      - 1..k hop (默认 k<=3) 的路径数，用于 S_route
      - 2..k hop 的路径数，用于 S_strength
      - 3-hop 且按 target 分开的路径数，用于 S_balance

    返回 DataFrame（按 Score(f) 降序）：
      fiber, nPos_all, nNeg_all, nPos_23, nNeg_23,
      nPos_3_t1, nNeg_3_t1, nPos_3_t2, nNeg_3_t2,
      S_route, S_strength, S_balance, Score
    """
    if df_counts is None or df_counts.empty:
        return pd.DataFrame(columns=[
            "fiber",
            "nPos_all", "nNeg_all",
            "nPos_23", "nNeg_23",
            "nPos_3_t1", "nNeg_3_t1",
            "nPos_3_t2", "nNeg_3_t2",
            "S_route", "S_strength", "S_balance", "Score"
        ])

    # preference for targets
    pref_t1 = pref_map.get(target1, "range")
    pref_t2 = pref_map.get(target2, "range")
    pref_by_target = {target1: pref_t1, target2: pref_t2}

    tmp = df_counts.copy()
    tmp["preference"] = tmp["target"].map(lambda t: pref_by_target.get(t, "range"))
    tmp["polarity"] = tmp.apply(lambda r: map_evidence_to_posnegunc(r["lastType"], r["preference"]), axis=1)

    # Only pos/neg matter for Eqs. (17)-(20) (range/uncertain will be excluded from counts)
    tmp = tmp[tmp["polarity"].isin(["pos", "neg"])].copy()

    if tmp.empty:
        return pd.DataFrame(columns=[
            "fiber",
            "nPos_all", "nNeg_all",
            "nPos_23", "nNeg_23",
            "nPos_3_t1", "nNeg_3_t1",
            "nPos_3_t2", "nNeg_3_t2",
            "S_route", "S_strength", "S_balance", "Score"
        ])

    # Aggregate counts
    # all hops (1..k), both targets
    all_agg = (
        tmp.groupby(["fiber", "polarity"])["nPaths"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={"pos": "nPos_all", "neg": "nNeg_all"})
        .reset_index()
    )
    if "nPos_all" not in all_agg.columns:
        all_agg["nPos_all"] = 0
    if "nNeg_all" not in all_agg.columns:
        all_agg["nNeg_all"] = 0

    # 2-3 hop subset
    tmp_23 = tmp[tmp["hop"].isin([2, 3])].copy()
    agg_23 = (
        tmp_23.groupby(["fiber", "polarity"])["nPaths"]
        .sum()
        .unstack(fill_value=0)
        .rename(columns={"pos": "nPos_23", "neg": "nNeg_23"})
        .reset_index()
    )
    if agg_23.empty:
        agg_23 = pd.DataFrame({"fiber": all_agg["fiber"], "nPos_23": 0, "nNeg_23": 0})
    else:
        if "nPos_23" not in agg_23.columns:
            agg_23["nPos_23"] = 0
        if "nNeg_23" not in agg_23.columns:
            agg_23["nNeg_23"] = 0

    # 3-hop target-wise
    tmp_3 = tmp[tmp["hop"] == 3].copy()
    agg_3 = (
        tmp_3.groupby(["fiber", "target", "polarity"])["nPaths"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("pos", "neg"):
        if col not in agg_3.columns:
            agg_3[col] = 0

    def pick_target_counts(df, tgt, pos_col_name, neg_col_name):
        sub = df[df["target"] == tgt][["fiber", "pos", "neg"]].copy()
        sub = sub.rename(columns={"pos": pos_col_name, "neg": neg_col_name})
        return sub

    t1_3 = pick_target_counts(agg_3, target1, "nPos_3_t1", "nNeg_3_t1")
    t2_3 = pick_target_counts(agg_3, target2, "nPos_3_t2", "nNeg_3_t2")

    # Merge all pieces
    out = all_agg.merge(agg_23, on="fiber", how="left")
    out = out.merge(t1_3, on="fiber", how="left").merge(t2_3, on="fiber", how="left")
    for col in ["nPos_23", "nNeg_23", "nPos_3_t1", "nNeg_3_t1", "nPos_3_t2", "nNeg_3_t2"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)

    # Eq. (17)
    alpha = float(alpha)
    out["S_route"] = (out["nPos_all"] + alpha) / (out["nPos_all"] + out["nNeg_all"] + 2 * alpha)

    # Eq. (18)
    out["S_strength"] = (out["nPos_23"] + alpha) / (out["nPos_23"] + out["nNeg_23"] + 2 * alpha)

    # Eq. (19)
    def net_support(pos, neg):
        # Stabilized net support from 3-hop counts (+1 in denom)
        denom = (pos + neg + 1.0)
        return (pos - neg) / denom if denom != 0 else 0.0

    s1 = out.apply(lambda r: net_support(r["nPos_3_t1"], r["nNeg_3_t1"]), axis=1)
    s2 = out.apply(lambda r: net_support(r["nPos_3_t2"], r["nNeg_3_t2"]), axis=1)
    out["S_balance"] = 1.0 - (s1 - s2).abs() / (s1.abs() + s2.abs() + 1.0)

    # Eq. (20)
    out["Score"] = out["S_route"] + out["S_strength"] + out["S_balance"]

    # Keep fibers with at least one effective (pos/neg) path toward either target
    out["nEff_all"] = out["nPos_all"] + out["nNeg_all"]
    out = out[out["nEff_all"] > 0].copy()

    out = out.drop(columns=["nEff_all"])
    out = out.sort_values(by=["Score", "S_route", "S_strength", "S_balance"], ascending=False).reset_index(drop=True)

    # Round for display
    for c in ["S_route", "S_strength", "S_balance", "Score"]:
        out[c] = out[c].astype(float).round(4)

    return out



# -----------------------------
# Table10-based path scoring (paper-reproducible)
#  - This reproduces Table C3 by computing pos/neg via sign propagation along the path:
#      net_effect = Π sign(rel_i), where sign(Increases)=+1, sign(Decreases)=-1, Causes is neutral
#    Then map net_effect to (pos/neg) by target desirability (higher/lower is better).
#  - Exact duplicate paths are removed by (fiber, target, hops, node_seq, rel_seq).
# -----------------------------
TABLE10_FILENAME = "Table10.xlsx"
TABLE10_SHEET = "Table10"

@st.cache_data(show_spinner=False)
def load_table10_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(base_dir, TABLE10_FILENAME)
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"Cannot find {TABLE10_FILENAME} in: {base_dir}\n"
            f"Please place {TABLE10_FILENAME} next to apptest.py."
        )
    df = pd.read_excel(xlsx_path, sheet_name=TABLE10_SHEET)
    need = {"fiber","target","hops","node_seq","rel_seq"}
    if not need.issubset(set(df.columns)):
        raise ValueError(
            f"{TABLE10_FILENAME}/{TABLE10_SHEET} must contain columns: {sorted(need)}. "
            f"Got: {list(df.columns)}"
        )

    def _parse_list_str(s):
        if isinstance(s, list):
            return s
        if not isinstance(s, str):
            return []
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [p.strip() for p in inner.split(",")]
        return [s]

    out = df.copy()
    out["hops"] = pd.to_numeric(out["hops"], errors="coerce").fillna(0).astype(int)
    out["node_list"] = out["node_seq"].apply(_parse_list_str)
    out["rel_list"] = out["rel_seq"].apply(_parse_list_str)

    # Drop exact duplicates (this is required to reproduce Table C3, e.g., duplicated basalt ductility paths)
    out = out.drop_duplicates(subset=["fiber","target","hops","node_seq","rel_seq"]).reset_index(drop=True)
    return out

def _net_effect_sign(rel_list: list[str]) -> int:
    """Return +1 (net increases target) or -1 (net decreases target) by multiplying signed relations."""
    s = 1
    for r in (rel_list or []):
        r = (r or "").strip()
        if r == "Increases":
            s *= 1
        elif r == "Decreases":
            s *= -1
        else:
            # Causes or other bridge relations treated as neutral
            s *= 1
    return 1 if s >= 0 else -1

def _target_preference(target_name: str, pref_map: dict) -> str:
    """Return 'higher'/'lower'/'range'. For reproducibility, hard-code MS as higher and MF as lower."""
    tl = (target_name or "").strip().lower()
    if "marshall stability" in tl or tl == "ms":
        return "higher"
    if "marshall flow" in tl or tl == "mf":
        return "lower"
    return (pref_map.get(target_name) or "range")

def compute_path_level_scores_from_table10(
    table10_df: pd.DataFrame,
    pref_map: dict,
    target1: str,
    target2: str,
    max_hops: int,
    alpha: float = 1.0,
):
    """Compute Table C3 style statistics from Table10.xlsx."""
    if table10_df is None or table10_df.empty:
        return pd.DataFrame(columns=["fiber","P_all","N_all","P_all_23","N_all_23","Delta_net_MS","Delta_net_MF","S_route","S_strength","S_balance","Score"])

    max_hops = int(max_hops)
    if max_hops < 1:
        max_hops = 1
    if max_hops > 3:
        max_hops = 3

    t1 = str(target1).strip()
    t2 = str(target2).strip()
    df = table10_df.copy()
    df = df[df["target"].isin([t1, t2]) & (df["hops"] >= 1) & (df["hops"] <= max_hops)].copy()
    if df.empty:
        return pd.DataFrame(columns=["fiber","P_all","N_all","P_all_23","N_all_23","Delta_net_MS","Delta_net_MF","S_route","S_strength","S_balance","Score"])

    # Compute net effect sign and map to pos/neg by target preference
    df["eff_sign"] = df["rel_list"].apply(_net_effect_sign)
    pref_t = {t1: _target_preference(t1, pref_map), t2: _target_preference(t2, pref_map)}

    def _pol_by_target(target, eff_sign):
        pref = pref_t.get(target, "range")
        if pref == "range":
            return "unc"
        # eff_sign +1 means net increases target; -1 means net decreases target
        if pref == "higher":
            return "pos" if eff_sign == 1 else "neg"
        # lower-is-better
        return "pos" if eff_sign == -1 else "neg"

    df["polarity"] = df.apply(lambda r: _pol_by_target(r["target"], int(r["eff_sign"])), axis=1)
    df = df[df["polarity"].isin(["pos","neg"])].copy()

    # Helper for stable column names
    def _short_target(t: str) -> str:
        tl = (t or "").lower()
        if "marshall stability" in tl or tl.strip() in {"ms"}:
            return "MS"
        if "marshall flow" in tl or tl.strip() in {"mf"}:
            return "MF"
        return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")[:16] or "T"

    t1s = _short_target(t1)
    t2s = _short_target(t2)

    alpha = float(alpha)

    rows = []
    for fiber, g in df.groupby("fiber"):
        # (optional) drop generic 'Fiber' bucket to match paper tables
        if str(fiber).strip().lower() == "fiber":
            continue

        P_all = int((g["polarity"] == "pos").sum())
        N_all = int((g["polarity"] == "neg").sum())

        g23 = g[g["hops"].isin([2,3])]
        P_all_23 = int((g23["polarity"] == "pos").sum())
        N_all_23 = int((g23["polarity"] == "neg").sum())

        g_t1 = g[g["target"] == t1]
        g_t2 = g[g["target"] == t2]
        d1 = int((g_t1["polarity"] == "pos").sum() - (g_t1["polarity"] == "neg").sum())
        d2 = int((g_t2["polarity"] == "pos").sum() - (g_t2["polarity"] == "neg").sum())

        S_route = (P_all + alpha) / (P_all + N_all + 2 * alpha) if (P_all + N_all + 2 * alpha) != 0 else 0.0
        S_strength = (P_all_23 + alpha) / (P_all_23 + N_all_23 + 2 * alpha) if (P_all_23 + N_all_23 + 2 * alpha) != 0 else 0.0
        S_balance = 1.0 - abs(d1 - d2) / (abs(d1) + abs(d2) + 1.0)
        # Paper-style rounding: round each sub-score to 2 decimals before summing
        S_route_r = round(S_route, 2)
        S_strength_r = round(S_strength, 2)
        S_balance_r = round(S_balance, 2)
        Score = S_route_r + S_strength_r + S_balance_r

        rows.append({
            "fiber": fiber,
            "P_all": P_all,
            "N_all": N_all,
            "P_all_23": P_all_23,
            "N_all_23": N_all_23,
            f"Delta_net_{t1s}": d1,
            f"Delta_net_{t2s}": d2,
            "S_route": round(float(S_route), 4),
            "S_strength": round(float(S_strength), 4),
            "S_balance": round(float(S_balance), 4),
            "Score": round(float(Score), 4),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(by=["Score","S_route","S_strength","S_balance"], ascending=False).reset_index(drop=True)
    return out


def _ensure_state(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


# UI states (shared across tabs)
_ensure_state("center_label", "Component")
_ensure_state("center_name", "Basalt fiber")
_ensure_state("k_hop", 3)
_ensure_state("max_paths_subgraph", 500)
_ensure_state("max_paths_global", 2000)
_ensure_state("target1", None)
_ensure_state("target2", None)
_ensure_state("k_path", 3)
_ensure_state("alpha_path", 1.0)
_ensure_state("max_distinct_paths", 20000)

# --- UI defaults migration (apply once) ---
# If you have old session_state values from previous runs (e.g., k_path=1 or alpha_path=0),
# this block updates them ONCE to the new defaults requested by the user.
if "ui_defaults_v2_applied" not in st.session_state:
    st.session_state["ui_defaults_v2_applied"] = True
    st.session_state["k_path"] = 3          # default Max hops = 3
    st.session_state["alpha_path"] = 1.0    # default alpha = 1
    st.session_state["max_distinct_paths"] = 1000  # fixed safety cap (no UI slider)


# --- UI defaults migration v3 (apply once) ---
# Force UI defaults requested later: center defaults + target defaults.
# (Targets are applied after property options are loaded.)
if "ui_defaults_v3_applied" not in st.session_state:
    st.session_state["ui_defaults_v3_applied"] = True
    st.session_state["center_label"] = "Component"
    st.session_state["center_name"] = "Basalt fiber"



# -----------------------------
# Property options for targets (loaded once)
# -----------------------------
try:
    _pref_map_for_ui, _mapping_df_for_ui = load_preference_map_from_excel()
    _PROP_OPTIONS_ALL = sorted(
        _mapping_df_for_ui["PropertyName"].dropna().astype(str).str.strip().unique().tolist()
    )
except Exception as _e:
    _PROP_OPTIONS_ALL = []
    st.error(f"Failed to load property list from {MAPPING_FILENAME}/{MAPPING_SHEET}: {_e}")


# --- UI defaults migration v3 for targets (apply once) ---
# Override target1/target2 defaults even if old session_state exists.
if "ui_defaults_v3_targets_applied" not in st.session_state:
    st.session_state["ui_defaults_v3_targets_applied"] = True
    if _PROP_OPTIONS_ALL:
        def _pick_default(name_like: str) -> str:
            # exact match first
            for v in _PROP_OPTIONS_ALL:
                if v == name_like:
                    return v
            # case-insensitive substring fallback
            low = name_like.lower()
            for v in _PROP_OPTIONS_ALL:
                if low in v.lower():
                    return v
            return _PROP_OPTIONS_ALL[0]

        st.session_state["target1"] = _pick_default("Marshall stability")
        st.session_state["target2"] = _pick_default("Marshall flow value")


def _default_index(opt_list, name_like: str) -> int:
    if not opt_list:
        return 0
    try:
        return opt_list.index(name_like)
    except ValueError:
        low = name_like.lower()
        for i, v in enumerate(opt_list):
            if low in v.lower():
                return i
        return 0


# Set default targets once (if not set yet)
if st.session_state.get("target1") is None and _PROP_OPTIONS_ALL:
    st.session_state["target1"] = _PROP_OPTIONS_ALL[_default_index(_PROP_OPTIONS_ALL, "Marshall stability")]
if st.session_state.get("target2") is None and _PROP_OPTIONS_ALL:
    st.session_state["target2"] = _PROP_OPTIONS_ALL[_default_index(_PROP_OPTIONS_ALL, "Marshall flow value")]



# -----------------------------
# Main layout (Tabs workflow)
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "1  Load subgraph",
    "2  Global analysis (3.3.2)",
    "3  Path analysis (3.3.3)",
])


with tab1:
    st.subheader("Load subgraph")

    ctrl = st.container(border=True)
    with ctrl:
        c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
        with c1:
            # Level-1: choose center node label
            st.selectbox(
                "Center node label",
                options=CENTER_LABEL_OPTIONS,
                key="center_label",
            )

            # Level-2: choose center node name under the selected label
            _opts = fetch_node_names_by_label(st.session_state.center_label)
            if not _opts:
                st.warning(f"No nodes found for label: {st.session_state.center_label}")
                st.selectbox("Center node name", options=[""], key="center_name")
            else:
                # keep current selection if still valid; otherwise fallback
                _cur = st.session_state.get("center_name", "")
                if _cur not in _opts:
                    st.session_state["center_name"] = "Basalt fiber" if "Basalt fiber" in _opts else _opts[0]
                st.selectbox("Center node name", options=_opts, key="center_name")
        with c2:
            st.slider("k-hop", min_value=1, max_value=5, key="k_hop")
        with c3:
            st.slider("Max paths", 50, 2000, step=50, key="max_paths_subgraph")
        with c4:
            load_clicked = st.button("Load subgraph", use_container_width=True, key="btn_load_subgraph")

    if load_clicked:
        with st.spinner("Querying Neo4j..."):
            nodes, edges = fetch_subgraph(
                st.session_state.center_name,
                st.session_state.k_hop,
                st.session_state.max_paths_subgraph,
            )
        st.session_state.nodes = nodes
        st.session_state.edges = edges
        st.session_state.rel_types = sorted({e["type"] for e in edges})
        # 默认不过滤：rel_selected 为空 => show all
        st.session_state.rel_selected = []
        # 清空下游缓存
        st.session_state.global_stats = None
        st.session_state.global_detail_df = None
        st.session_state.path_scores_df = None
        st.session_state.path_raw_counts_df = None

    nodes = st.session_state.nodes
    edges = st.session_state.edges

    if not (nodes and edges):
        st.info("Please load a subgraph first: choose a center node and k-hop, then click **Load subgraph**.")
    else:
        left, right = st.columns([2, 1], gap="large")
        with right:
            st.markdown("#### Subgraph overview")
            st.write(f"**Center:** {st.session_state.center_name}  |  **k-hop:** {st.session_state.k_hop}")
            st.write(f"Nodes: **{len(nodes)}**, Edges: **{len(edges)}**")

            # Relationship filter
            if st.session_state.rel_types:
                st.multiselect(
                    "Relationship filter (empty = show ALL)",
                    options=st.session_state.rel_types,
                    key="rel_selected",
                )
                st.caption("Tip: keep this empty to show all edges.")

            df_edges = pd.DataFrame(edges)
            if not df_edges.empty and "type" in df_edges.columns:
                st.markdown("##### Relation types (current subgraph)")
                vc = df_edges["type"].value_counts()
                df_rel = vc.rename_axis("type").reset_index(name="count")
                st.dataframe(df_rel, use_container_width=True, height=260)

        with left:
            st.markdown("#### Graph view")
            rel_allow = st.session_state.rel_selected  # 空 => show all
            net = build_pyvis(nodes, edges, st.session_state.center_name, rel_allow)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                html_path = tmp.name
            html = open(html_path, "r", encoding="utf-8").read()
            st.components.v1.html(html, height=780, scrolling=True)
            os.remove(html_path)


with tab2:
    st.subheader("Global analysis (3.3.2)")

    ctrl = st.container(border=True)
    with ctrl:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write(f"**Center:** {st.session_state.center_name}  |  **k:** {st.session_state.k_hop}")
        with c2:
            compute_clicked = st.button("Compute global stats", use_container_width=True, key="btn_global")

    if compute_clicked:
        try:
            pref_map, mapping_df = load_preference_map_from_excel()
        except Exception as e:
            st.session_state.global_stats = None
            st.session_state.global_detail_df = None
            st.error(f"Failed to load mapping from {MAPPING_FILENAME}/{MAPPING_SHEET}: {e}")
        else:
            with st.spinner("Computing global stats from Neo4j..."):
                ev = fetch_property_lasttype_evidence(
                    center_name=st.session_state.center_name,
                    k_hop=st.session_state.k_hop,
                    max_paths=2000,
                    traversal_types=TRAVERSAL_TYPES_FOR_STATS,
                )
                stats, detail_df = compute_global_property_level_stats(ev, pref_map)
            st.session_state.global_stats = stats
            st.session_state.global_detail_df = detail_df

    if st.session_state.global_stats is None:
        st.info("Click **Compute global stats** above to generate Nprop/Npos/Nneg/Nunc/netscore.")
    else:
        s = st.session_state.global_stats
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Nprop", s["Nprop"])
        m2.metric("Npos", s["Npos"])
        m3.metric("Nneg", s["Nneg"])
        m4.metric("Nunc", s["Nunc"])
        m5.metric("netscore", s["netscore"])
        st.caption("netscore = (Npos - Nneg) / (Npos + Nneg + 2)  (Laplace smoothing +2)")

        detail = st.session_state.global_detail_df
        st.markdown("#### Property-level polarity details")
        st.dataframe(detail, use_container_width=True, height=420)

        st.download_button(
            "Download global details (CSV)",
            detail.to_csv(index=False),
            file_name=f"{st.session_state.center_name}_global_property_polarity_k{st.session_state.k_hop}.csv",
        )

        with st.expander("Show built-in polarity mapping table", expanded=False):
            try:
                _pref_map, _mapping_df = load_preference_map_from_excel()
                st.dataframe(_mapping_df, use_container_width=True, height=420)  # 全宽+更协调的高度
            except Exception as _e:
                st.error(f"Failed to load mapping table: {_e}")

with tab3:
    st.subheader("Path analysis (3.3.3)")

    ctrl = st.container(border=True)
    with ctrl:
        c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
        with c1:
            st.selectbox("Target 1", options=_PROP_OPTIONS_ALL, key="target1")
        with c2:
            st.selectbox("Target 2", options=_PROP_OPTIONS_ALL, key="target2")
        with c3:
            st.slider("Max hops", min_value=1, max_value=3, key="k_path")
        with c4:
            st.number_input("α (Laplace smoothing)", min_value=0.0, step=0.5, key="alpha_path")
        with c5:
            st.markdown("<div style=\'height: 28px\'></div>", unsafe_allow_html=True)
            compute_path_clicked = st.button("Compute Top-10", use_container_width=True, key="btn_path")

        # Fixed: Max distinct paths (safety cap)
        st.session_state["max_distinct_paths"] = 1000

    st.write(
        f"**Targets:** {st.session_state.target1}  &  {st.session_state.target2}  |  "
        f"**max hops:** {st.session_state.k_path}  |  **α:** {st.session_state.alpha_path}"
    )

    if compute_path_clicked:
        try:
            pref_map, _mapping_df = load_preference_map_from_excel()
        except Exception as e:
            st.session_state.path_scores_df = None
            st.session_state.path_raw_counts_df = None
            st.error(f"Failed to load mapping from {MAPPING_FILENAME}/{MAPPING_SHEET}: {e}")
        else:
            targets = [st.session_state.target1, st.session_state.target2]
            with st.spinner("Computing path-level scores from Neo4j (Top-10)..."):
                df_paths_all = load_table10_paths()
                # Keep a filtered view for raw-table inspection
                df_paths = df_paths_all[
                    df_paths_all["target"].isin(targets)
                    & (df_paths_all["hops"] >= 1)
                    & (df_paths_all["hops"] <= int(st.session_state.k_path))
                ].copy()

                df_scores = compute_path_level_scores_from_table10(
                    table10_df=df_paths_all,
                    pref_map=pref_map,
                    target1=st.session_state.target1,
                    target2=st.session_state.target2,
                    max_hops=st.session_state.k_path,
                    alpha=st.session_state.alpha_path,
                )
            st.session_state.path_raw_counts_df = df_paths
            st.session_state.path_scores_df = df_scores.head(10).copy()

    if st.session_state.path_scores_df is None:
        st.info("Click **Compute Top-10** above to generate a Table C3-style Top-10 result.")
    else:
        top10 = st.session_state.path_scores_df
        show_cols = None
        # Table10-based scorer returns Table C3-style columns:
        # fiber, P_all, N_all, P_all_23, N_all_23, Delta_net_*, S_route, S_strength, S_balance, Score
        delta_cols = [c for c in top10.columns if c.startswith("Delta_net_")]
        # Keep Delta_net columns in a stable order (MS first if present, then MF, then others)
        def _delta_sort_key(c: str):
            cl = c.lower()
            if cl.endswith("_ms") or cl == "delta_net_ms":
                return (0, c)
            if cl.endswith("_mf") or cl == "delta_net_mf":
                return (1, c)
            return (2, c)
        delta_cols = sorted(delta_cols, key=_delta_sort_key)

        show_cols = ["fiber", "P_all", "N_all", "P_all_23", "N_all_23"] + delta_cols + ["S_route", "S_strength", "S_balance", "Score"]
        # Safety: only keep columns that exist to avoid KeyError
        show_cols = [c for c in show_cols if c in top10.columns]

        st.markdown("#### Table C3 style (Top-10 fibers)")
        # ---- display formatting: keep integers as-is, non-integers -> 2 decimals ----
        display_df = top10[show_cols].copy()


        def _fmt_cell(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            # pandas / numpy ints
            if isinstance(v, (int,)):
                return str(v)
            # floats
            if isinstance(v, float):
                if math.isfinite(v) and abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
                return f"{v:.2f}"
            # fallback
            return str(v)


        # only format numeric columns; others (fiber) keep original
        for c in display_df.columns:
            if pd.api.types.is_numeric_dtype(display_df[c]):
                display_df[c] = display_df[c].map(_fmt_cell)

        st.dataframe(display_df, use_container_width=True, height=420)

        st.download_button(
            "Download Top-10 (CSV)",
            top10[show_cols].to_csv(index=False),
            file_name=f"Top10_path_score_{st.session_state.target1}_{st.session_state.target2}_k{st.session_state.k_path}.csv",
            key="dl_top10_csv",
        )

        with st.expander("Show raw path-count table (deduped)", expanded=False):
            raw = st.session_state.path_raw_counts_df
            st.dataframe(raw, use_container_width=True, height=320)
