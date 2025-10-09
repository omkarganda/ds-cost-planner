
# Startup Suitability Score — Streamlit App

This app demonstrates a practical, explainable pipeline to compute a 0–100 **Suitability Score** for a startup using a weighted scorecard, with red-flag gating, missingness and evidence coverage adjustments, and optional AHP weighting for better expert-derived pillar weights.

## Quick start (Streamlit Cloud)
1. Create a new app on Streamlit Cloud and point it to `app.py` in this repo.
2. Ensure `requirements.txt` contains `streamlit`, `pandas`, and `numpy`.
3. Deploy. Open the app and:
   - Load or edit weights in **Weights & Editing**.
   - Enter use-case scores in **Score a Startup**.
   - Adjust **red flags** and **coverage** in the sidebar.
   - (Optional) Use **AHP Weighting** to compute pillar weights from pairwise comparisons.

## File format for weights upload
CSV or Excel with columns:
- `Pillar`
- `Pillar Weight %`
- `Use Case`
- `Use Case Weight % (within pillar)`

Use-case weights are normalized **within each pillar** to sum to 100.

## Why this approach
- **Simple & transparent** for stakeholders today.
- **Gates critical risks** (founder severe hits auto-fail).
- **Handles missingness & coverage** so incomplete profiles aren’t overconfident.
- **Upgradable to ML**: later, swap the weighted sum with a trained model’s probability without changing the UX.
