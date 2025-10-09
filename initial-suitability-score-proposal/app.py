
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Startup Suitability Score", page_icon="📊", layout="wide")

# -------------------------------
# Helpers
# -------------------------------

DEFAULT_PILLARS = [
    # pillar_name, pillar_weight_pct (by stage), use_cases: list of (use_case_name, seed_weight_hint)
    ("Experience, Education & Execution", {"Pre-Seed": 40, "Seed": 30, "Series A+": 20},
     [("Prior Experience (domain-aligned)", 15),
      ("Domain Expertise (education/certs/practice)", 10),
      ("Execution Track Record (past ventures/outcomes)", 10),
      ("Education & Professional License (if applicable)", 5)]),
    ("Team Strength & Dynamics", {"Pre-Seed": 25, "Seed": 20, "Series A+": 15},
     [("Team Cohesion / History working together", 10),
      ("Equity Splits & Interpersonal Dynamics", 5),
      ("Hiring Capability (recent hiring velocity/quality)", 10)]),
    ("Background Check (Legal & Compliance)", {"Pre-Seed": 20, "Seed": 20, "Series A+": 20},
     [("Adverse Legal/Regulatory/PEP/Sanctions (pass if none/adjudicated)", 20)]),
    ("Ecosystem & Network", {"Pre-Seed": 15, "Seed": 10, "Series A+": 5},
     [("Advisory Network (credibility of advisors)", 10),
      ("External Support (accelerators/grants/partnerships)", 5)]),
]

def build_default_weights(stage: str) -> pd.DataFrame:
    rows = []
    for pillar_name, stage_weights, use_cases in DEFAULT_PILLARS:
        pillar_w = stage_weights.get(stage, list(stage_weights.values())[0])
        # normalize use-case hints within pillar to sum 100
        hints = np.array([w for _, w in use_cases], dtype=float)
        if hints.sum() == 0:
            uc_weights = np.ones_like(hints) / len(hints) * 100.0
        else:
            uc_weights = (hints / hints.sum()) * 100.0
        for (uc_name, hint), uc_w in zip(use_cases, uc_weights):
            rows.append({
                "Pillar": pillar_name,
                "Pillar Weight %": float(pillar_w),
                "Use Case": uc_name,
                "Use Case Weight % (within pillar)": float(round(uc_w, 2))
            })
    df = pd.DataFrame(rows)
    return df

def normalize_pillar_weights(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure pillar weights sum to 100; rescale proportionally if not
    out = df.copy()
    pw = out[["Pillar", "Pillar Weight %"]].drop_duplicates()
    total = pw["Pillar Weight %"].sum()
    if total <= 0:
        return out
    scale = 100.0 / total
    out["Pillar Weight %"] = out["Pillar Weight %"] * scale
    return out

def compute_suitability(df_weights: pd.DataFrame, uc_scores: dict, founder_redflag: bool,
                        cofounder_redflag: bool, cofounder_penalty: float,
                        evidence_coverage_pct: float, missingness_penalty: float):
    """
    df_weights columns: Pillar, Pillar Weight %, Use Case, Use Case Weight % (within pillar)
    uc_scores: dict mapping Use Case -> 0..100 or None
    founder_redflag: if True, auto fail (0)
    cofounder_redflag: if True, apply multiplicative penalty (e.g., 0.85)
    evidence_coverage_pct: 0..100 coverage of verified evidence provided
    missingness_penalty: 0..1 multiplier applied to fraction of missing signals
    """
    # Red-flag: founder hard fail
    if founder_redflag:
        return 0.0, {"reason": "Founder failed hard red-flag (e.g., SSN trace/criminal). Auto-unsuitable."}, 0.0, {}

    # Compute base weighted score
    df = df_weights.copy()
    # Normalize pillar weights to sum 100
    df = normalize_pillar_weights(df)
    # Normalize use-case weights within each pillar to sum 100
    df["Use Case Weight % (within pillar)"] = df.groupby("Pillar")["Use Case Weight % (within pillar)"].transform(
        lambda s: (s / s.sum()) * 100.0 if s.sum() > 0 else s
    )

    # Merge scores
    df["Use Case Score"] = df["Use Case"].map(lambda k: uc_scores.get(k, None))

    # Missingness
    n_total = len(df)
    n_missing = df["Use Case Score"].isna().sum()
    missing_fraction = n_missing / n_total if n_total > 0 else 0.0

    # For missing scores, treat as neutral (50) but apply penalty later; or leave as None -> compute with available weights scaled
    # Here: compute pillar sub-score using available use-cases only
    def weighted_avg(sub):
        sub_avail = sub.dropna(subset=["Use Case Score"])
        if len(sub_avail) == 0:
            return np.nan
        # renormalize within pillar for available UCs
        w = sub_avail["Use Case Weight % (within pillar)"].values
        w = w / w.sum()
        x = sub_avail["Use Case Score"].values / 100.0
        return (w @ x) * 100.0

    pillar_scores = df.groupby(["Pillar", "Pillar Weight %"], as_index=False).apply(weighted_avg).reset_index()
    pillar_scores.columns = ["Pillar", "Pillar Weight %", "score"]
    pillar_scores["score"] = pillar_scores["score"].astype(float)

    # Compute overall weighted sum using available pillars only, renormalize pillar weights among available
    avail = pillar_scores.dropna(subset=["score"]).copy()
    if len(avail) == 0:
        base = 0.0
    else:
        w = avail["Pillar Weight %"].values
        w = w / w.sum()
        x = avail["score"].values / 100.0
        base = (w @ x) * 100.0

    # Apply co-founder penalty if any
    if cofounder_redflag:
        base *= cofounder_penalty  # e.g., 0.85

    # Apply missingness penalty (penalize unexplained gaps)
    base *= (1.0 - missingness_penalty * missing_fraction)

    # Apply evidence coverage attenuation: if only 60% of evidence verified, don't allow score to exceed coverage
    coverage = max(0.0, min(1.0, evidence_coverage_pct / 100.0))
    base = min(base, coverage * 100.0)

    # Naive "confidence" proxy from coverage & completeness
    confidence = (1.0 - missing_fraction) * coverage

    # Build explanations
    drivers = {}
    for _, row in avail.iterrows():
        drivers[row["Pillar"]] = round(float(row["score"]), 1)

    meta = {
        "missing_fraction": round(missing_fraction, 2),
        "coverage": coverage,
    }
    return float(round(base, 2)), meta, float(round(confidence, 2)), drivers

def ahp_weights(pillars, matrix):
    """
    Compute AHP weights from a pairwise comparison matrix (Saaty scale).
    pillars: list of pillar names
    matrix: numpy array (n x n)
    Returns: pd.DataFrame with Pillar, AHP Weight %
    """
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_idx = np.argmax(np.real(eigvals))
    w = np.real(eigvecs[:, max_idx])
    w = np.abs(w)
    w = w / w.sum()
    df = pd.DataFrame({"Pillar": pillars, "AHP Weight %": (w * 100.0)})
    return df

# -------------------------------
# Sidebar Controls
# -------------------------------

st.sidebar.title("⚙️ Controls")

stage = st.sidebar.selectbox("Company Stage", ["Pre-Seed", "Seed", "Series A+"], index=0)

st.sidebar.markdown("### Red Flags")
founder_redflag = st.sidebar.checkbox("Founder fails hard red-flag (auto unsuitability)", value=False,
                                      help="E.g., SSN Trace fail, severe criminal record, sanctions, fraud")
cofounder_redflag = st.sidebar.checkbox("Co-founder fails hard red-flag", value=False,
                                        help="Not auto-fail, but penalizes the score")
cofounder_penalty = st.sidebar.slider("Co-founder penalty (multiplier)", 0.5, 1.0, 0.85, 0.01)

st.sidebar.markdown("### Evidence & Missingness")
evidence_coverage_pct = st.sidebar.slider("Verified evidence coverage (%)", 0, 100, 80, 5,
                                          help="Percent of required evidence verified at source of truth")
missingness_penalty = st.sidebar.slider("Missingness penalty strength", 0.0, 0.8, 0.3, 0.05,
                                        help="How strongly to penalize missing sub-scores")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Founder failure = auto-unsuitable; Co-founder failure applies a partial penalty.\n"
                   "Coverage caps the score; missingness reduces it.")

# -------------------------------
# Main Layout
# -------------------------------

st.title("📊 Startup Suitability Score (Demo)")
st.caption("Start with a weighted model, add rules for red flags and missingness, optional AHP for better weights, and a roadmap to ML.")

tab_overview, tab_weights, tab_score, tab_ahp, tab_roadmap = st.tabs(
    ["Overview", "Weights & Editing", "Score a Startup", "Optional: AHP Weighting", "Roadmap & Docs"]
)

# -------------------------------
# Overview
# -------------------------------
with tab_overview:
    st.subheader("What this app does")
    st.write("""
    **Goal:** Produce a 0–100 *Suitability Score* representing the startup's likelihood of success using verified inputs.
    
    **How it works (today):**
    1. Use a **weighted scorecard** across pillars like Team, Experience, Background, Ecosystem.
    2. Enforce **red-flag rules** (e.g., founder severe criminal hit ⇒ auto-unsuitable).
    3. Apply **penalties for missing data** and cap the score by **evidence coverage**.
    4. Optionally, compute pillar weights via **AHP pairwise comparisons** (more consistent expert weights).
    
    **Why weighted scorecards are good:** simple, transparent, easy to adjust.
    
    **Limitations:** weights can be subjective; small changes can swing results; pure sums can miss hard deal-breakers; stage differences matter.
    """)

    st.markdown("#### Enhancements included here")
    st.markdown("""
    - **Red-flag gating:** Founder failure ⇒ score = 0; co-founder failure ⇒ configurable penalty.  
    - **Missingness handling:** If some sub-scores are unknown, we use available signals and apply a penalty based on how much is missing.  
    - **Evidence coverage cap:** If only 60% of evidence is verified, the score won't exceed 60.  
    - **AHP (optional):** Replace manual pillar weights with pairwise-comparison-derived weights.
    """)
    st.markdown("#### How to use (quick demo)")
    st.markdown("""
    1. Go to **Weights & Editing** to review/edit pillar and use-case weights for the selected stage.  
    2. Go to **Score a Startup** and input 0–100 scores for each use case (e.g., team cohesion 80).  
    3. Set any **red flags** and **coverage** in the left sidebar.  
    4. Read the **explanations** and **drivers** shown with the final score.  
    5. (Optional) Use **AHP Weighting** to compute expert-driven pillar weights via pairwise comparisons.
    """)

# -------------------------------
# Weights & Editing
# -------------------------------
with tab_weights:
    st.subheader("Pillar & Use-Case Weights")
    st.caption("Pillar weights are stage-specific. Use-case weights are normalized **within pillar** to 100%.")

    # Allow upload of a weights file, otherwise use defaults
    uploaded = st.file_uploader("Upload weights (CSV or Excel)", type=["csv", "xlsx"])
    if "weights_df" not in st.session_state:
        st.session_state["weights_df"] = build_default_weights(stage)

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            # Basic column normalization
            expected_cols = ["Pillar", "Pillar Weight %", "Use Case", "Use Case Weight % (within pillar)"]
            # Best-effort rename
            rename_map = {}
            for col in df_up.columns:
                low = col.strip().lower()
                if "pillar weight" in low and "%" in low:
                    rename_map[col] = "Pillar Weight %"
                elif "pillar" == low or low.startswith("pillar"):
                    rename_map[col] = "Pillar"
                elif "use case weight" in low:
                    rename_map[col] = "Use Case Weight % (within pillar)"
                elif "use case" in low:
                    rename_map[col] = "Use Case"
            df_up = df_up.rename(columns=rename_map)
            missing = [c for c in expected_cols if c not in df_up.columns]
            if missing:
                st.warning(f"Uploaded file missing columns: {missing}. Using defaults instead.")
            else:
                st.session_state["weights_df"] = df_up[expected_cols].copy()
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    # Stage switch resets defaults (optional)
    if st.button("Reset to Stage Defaults"):
        st.session_state["weights_df"] = build_default_weights(stage)

    # Show editable data
    st.markdown("##### Edit Weights (click cells to edit)")
    edited = st.data_editor(st.session_state["weights_df"], num_rows="dynamic", use_container_width=True,
                            key=f"editor_{stage}")
    # Store back
    st.session_state["weights_df"] = edited

    # Display normalized pillar weights summary
    norm_df = normalize_pillar_weights(edited)
    pw = norm_df[["Pillar", "Pillar Weight %"]].drop_duplicates().sort_values("Pillar Weight %", ascending=False)
    st.markdown("##### Pillar Weights (normalized to sum 100%)")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(pw, use_container_width=True)
    with col2:
        st.metric("Sum of Pillar Weights", f"{pw['Pillar Weight %'].sum():.2f}%")

# -------------------------------
# Score a Startup
# -------------------------------
with tab_score:
    st.subheader("Enter Use-Case Scores (0–100)")
    st.caption("Use simple 0–100 sliders to rate the startup on each use case. Leave as-is if unknown; missingness penalty will apply.")

    dfw = st.session_state["weights_df"].copy()
    dfw = normalize_pillar_weights(dfw)

    uc_scores = {}
    missing_tracker = {}

    for pillar, sub in dfw.groupby("Pillar"):
        st.markdown(f"**{pillar}**")
        cols = st.columns(2)
        with cols[0]:
            st.write("Use Case")
        with cols[1]:
            st.write("Score (0–100)")
        for _, row in sub.iterrows():
            uc = row["Use Case"]
            # Slider with default None is not supported; we simulate missing by a checkbox
            with st.container():
                c1, c2, c3 = st.columns([3, 2, 1])
                c1.write(f"- {uc}")
                known = c3.checkbox("known", value=True, key=f"known_{pillar}_{uc}")
                if known:
                    val = c2.slider("", 0, 100, 70, key=f"score_{pillar}_{uc}")
                    uc_scores[uc] = val
                    missing_tracker[uc] = False
                else:
                    uc_scores[uc] = None
                    missing_tracker[uc] = True

    score, meta, confidence, drivers = compute_suitability(
        dfw, uc_scores,
        founder_redflag, cofounder_redflag, cofounder_penalty,
        evidence_coverage_pct, missingness_penalty
    )

    st.markdown("---")
    st.metric("Final Suitability Score", f"{score:.2f} / 100")
    st.progress(min(1.0, score/100.0))

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Confidence (proxy)", f"{confidence:.2f}")
    with colB:
        st.metric("Missing Fraction", f"{meta.get('missing_fraction', 0):.2f}")
    with colC:
        st.metric("Verified Coverage", f"{int(meta.get('coverage', 0)*100)}%")

    st.markdown("##### Top Drivers (pillar-level sub-scores)")
    if drivers:
        dd = pd.DataFrame([{"Pillar": k, "Pillar Sub-Score": v} for k, v in drivers.items()])
        dd = dd.sort_values("Pillar Sub-Score", ascending=False)
        st.dataframe(dd, use_container_width=True)
    else:
        st.info("No available use-case scores to compute drivers.")

    # Explanation block
    st.markdown("##### Explanation")
    expl = []
    if founder_redflag:
        expl.append("Founder failed hard red-flag ⇒ auto-unsuitable.")
    if cofounder_redflag:
        expl.append(f"Co-founder red-flag penalty applied (×{cofounder_penalty:.2f}).")
    if meta.get("missing_fraction", 0) > 0:
        expl.append(f"Missingness penalty applied based on {meta.get('missing_fraction', 0):.2f} fraction missing.")
    if meta.get("coverage", 1.0) < 1.0:
        expl.append(f"Evidence coverage cap applied at {int(meta.get('coverage', 0)*100)}%.")
    if not expl:
        expl.append("Score is a weighted sum of pillar sub-scores; no penalties applied.")
    st.write(" • " + "\n • ".join(expl))

# -------------------------------
# AHP Weighting
# -------------------------------
with tab_ahp:
    st.subheader("Optional: Compute Pillar Weights with AHP")
    st.caption("Use pairwise comparisons (Saaty scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme). "
               "Matrix must be reciprocal (if A vs B = 3, then B vs A = 1/3).")

    # Get current pillar list
    current_pw = st.session_state["weights_df"][["Pillar", "Pillar Weight %"]].drop_duplicates()
    pillars = current_pw["Pillar"].tolist()
    n = len(pillars)
    if n == 0:
        st.warning("No pillars found.")
    else:
        st.write("Pillars:", ", ".join(pillars))

        # Initialize a default pairwise matrix where all equal importance (1s)
        if "ahp_matrix" not in st.session_state or st.session_state.get("ahp_pillars", []) != pillars:
            mat = np.ones((n, n), dtype=float)
            st.session_state["ahp_matrix"] = mat
            st.session_state["ahp_pillars"] = pillars

        # Editable table-like inputs
        mat = st.session_state["ahp_matrix"].copy()
        for i in range(n):
            cols = st.columns(n+1)
            cols[0].markdown(f"**{pillars[i]}**")
            for j in range(n):
                if i == j:
                    cols[j+1].number_input("", value=1.0, disabled=True, key=f"ahp_{i}_{j}")
                    mat[i, j] = 1.0
                elif i < j:
                    val = cols[j+1].number_input("", value=float(mat[i, j]), key=f"ahp_{i}_{j}", step=0.1)
                    mat[i, j] = float(val)
                else:
                    # reciprocal
                    val = 1.0 / float(st.session_state.get(f"ahp_{j}_{i}", mat[j, i]))
                    cols[j+1].number_input("", value=float(val), disabled=True, key=f"ahp_{i}_{j}")
                    mat[i, j] = val

        st.session_state["ahp_matrix"] = mat
        try:
            ahp_df = ahp_weights(pillars, mat)
            st.markdown("##### AHP-derived Pillar Weights")
            st.dataframe(ahp_df, use_container_width=True)
            if st.button("Apply AHP Weights to Current Stage"):
                # Map into weights_df
                wdf = st.session_state["weights_df"].copy()
                wdf = wdf.merge(ahp_df[["Pillar", "AHP Weight %"]], on="Pillar", how="left")
                wdf.loc[~wdf["AHP Weight %"].isna(), "Pillar Weight %"] = wdf["AHP Weight %"]
                wdf = wdf.drop(columns=["AHP Weight %"])
                st.session_state["weights_df"] = wdf
                st.success("Applied AHP weights to the current weights table.")
        except Exception as e:
            st.error(f"Failed to compute AHP weights: {e}")

# -------------------------------
# Roadmap & Docs
# -------------------------------
with tab_roadmap:
    st.subheader("Roadmap & Notes")
    st.markdown("""
    **Today (MVP):** Weighted scoring + red-flag gating + missingness & coverage adjustments.  
    **Near-term:** Back-test weights against past decisions; refine; add human-in-loop overrides.  
    **Mid-term:** Train a ML model (e.g., logistic regression, gradient boosted trees) on verified features to predict success probability; report calibration (Brier score, reliability curves).  
    **Long-term:** Add uncertainty quantification (conformal intervals), fairness monitoring, and a full audit log (versioned features, rationale, evidence).
    
    **Why this structure works:**  
    - It’s explainable for stakeholders today.  
    - It accommodates strict compliance rules (red flags).  
    - It creates a clean upgrade path to ML without changing the user experience.
    """)

    st.markdown("#### Example: Mapping Verified Inputs to Use-Case Scores")
    st.write("""
    - *Prior Experience (domain-aligned):* 0–100 based on years in domain roles and relevance of past roles to current venture.  
    - *Domain Expertise:* 0–100 based on degrees/certs or ≥5 years applied experience + publications/patents.  
    - *Team Cohesion:* 0–100 based on documented co-working history and low early attrition.  
    - *Legal/Compliance (Adverse findings):* 0–100 where 100 = no adverse hits; 0 = severe unmitigated findings (but severe founder hit is a red-flag ⇒ auto 0).  
    - *Advisory Network:* 0–100 based on advisor track record and relevance.
    """)

    st.markdown("#### Notes for Demo")
    st.write("""
    - You can **upload** a weights file (CSV/XLSX) with columns:
      `Pillar, Pillar Weight %, Use Case, Use Case Weight % (within pillar)`  
    - Or click **Reset to Stage Defaults** to load a sensible template (from your internal doc).  
    - Use the left **sidebar** to simulate red flags and coverage.  
    - Use **AHP tab** to compute expert weights from pairwise inputs.
    """)
