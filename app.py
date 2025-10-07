# Re-create Streamlit app files and a zip bundle for download

import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="DS Platform Cost Planner (Stakeholder Edition)",
    layout="wide",
    page_icon="💸"
)

# ---------- Helpers ----------
def fmt(n):
    try:
        return "${:,.2f}".format(float(n))
    except Exception:
        return str(n)

def apply_preset(name: str):
    """Return default values for a named preset."""
    if name == "Lean MVP":
        return dict(
            monthly_requests=20000, seconds_per_request=2.0, serverless_mem_gb=2.0, logs_gb=10,
            sm_notebook_hours=80, sm_train_hours=8, sm_pipe_hours=30,
            vx_endpoint_nodes=1, vx_workbench_hours=80, vx_train_hours=8, vx_pipe_hours=30,
            dbx_nb_hours=80, dbx_job_hours=8, dbx_pipe_hours=30, dbx_serve_hours=0,
        )
    if name == "Pilot customer":
        return dict(
            monthly_requests=100000, seconds_per_request=2.0, serverless_mem_gb=2.0, logs_gb=20,
            sm_notebook_hours=120, sm_train_hours=10, sm_pipe_hours=60,
            vx_endpoint_nodes=1, vx_workbench_hours=120, vx_train_hours=10, vx_pipe_hours=60,
            dbx_nb_hours=120, dbx_job_hours=10, dbx_pipe_hours=60, dbx_serve_hours=0,
        )
    if name == "10× scale":
        return dict(
            monthly_requests=200000, seconds_per_request=2.0, serverless_mem_gb=2.5, logs_gb=40,
            sm_notebook_hours=160, sm_train_hours=16, sm_pipe_hours=120,
            vx_endpoint_nodes=2, vx_workbench_hours=160, vx_train_hours=16, vx_pipe_hours=120,
            dbx_nb_hours=160, dbx_job_hours=16, dbx_pipe_hours=120, dbx_serve_hours=0,
        )
    # fallback
    return dict(
        monthly_requests=20000, seconds_per_request=2.0, serverless_mem_gb=2.0, logs_gb=10,
        sm_notebook_hours=80, sm_train_hours=8, sm_pipe_hours=30,
        vx_endpoint_nodes=1, vx_workbench_hours=80, vx_train_hours=8, vx_pipe_hours=30,
        dbx_nb_hours=80, dbx_job_hours=8, dbx_pipe_hours=30, dbx_serve_hours=0,
    )

def ensure_defaults():
    if "initialized" not in st.session_state:
        # static default rates (list-price style; editable in Advanced)
        st.session_state.update(dict(
            # Shared
            monthly_requests=20000, seconds_per_request=2.0, serverless_mem_gb=2.0, logs_gb=10,

            # SageMaker (rates)
            sm_rate_per_sec_gb=0.0000200,
            sm_notebook_price_per_hour=0.23,
            sm_train_price_per_hour=0.461,
            sm_pipe_price_per_hour=0.23,
            sm_s3_price_per_gb=0.023,
            sm_logs_price_per_gb=0.50,
            # SageMaker (hours/storage)
            sm_notebook_hours=80,
            sm_train_hours=8,
            sm_pipe_hours=30,
            sm_storage_gb=200,

            # Vertex AI (rates)
            vx_endpoint_hours=730,
            vx_endpoint_price_per_hour=0.23,
            vx_workbench_price_per_hour=0.23,
            vx_train_price_per_hour=0.46,
            vx_pipe_price_per_hour=0.23,
            vx_gcs_price_per_gb=0.02,
            vx_logs_price_per_gb=0.50,
            # Vertex AI (hours/storage & nodes)
            vx_endpoint_nodes=1,
            vx_workbench_hours=80,
            vx_train_hours=8,
            vx_pipe_hours=30,
            vx_storage_gb=200,

            # Databricks (rates)
            dbx_nb_dbus_per_hour=3.0,
            dbx_nb_price_per_dbu=0.55,
            dbx_nb_ec2_per_hour=0.192,
            dbx_job_dbus_per_hour=3.0,
            dbx_job_price_per_dbu=0.30,
            dbx_job_ec2_per_hour=0.384,
            dbx_pipe_dbus_per_hour=3.0,
            dbx_pipe_price_per_dbu=0.30,
            dbx_pipe_ec2_per_hour=0.192,
            dbx_serve_dbus_per_hour=5.0,
            dbx_serve_price_per_dbu=0.30,
            dbx_serve_ec2_per_hour=0.0,
            dbx_s3_price_per_gb=0.023,
            dbx_logs_price_per_gb=0.50,
            # Databricks (hours/storage)
            dbx_nb_hours=80,
            dbx_job_hours=8,
            dbx_pipe_hours=30,
            dbx_serve_hours=0,
            dbx_storage_gb=200,
        ))
        st.session_state.initialized = True

def sagemaker_cost(ss):
    inference = ss.monthly_requests * ss.seconds_per_request * ss.serverless_mem_gb * ss.sm_rate_per_sec_gb
    notebooks = ss.sm_notebook_hours * ss.sm_notebook_price_per_hour
    training = ss.sm_train_hours * ss.sm_train_price_per_hour
    pipelines = ss.sm_pipe_hours * ss.sm_pipe_price_per_hour
    storage = ss.sm_storage_gb * ss.sm_s3_price_per_gb
    logs = ss.logs_gb * ss.sm_logs_price_per_gb
    total = inference + notebooks + training + pipelines + storage + logs
    unit = (inference / ss.monthly_requests) if ss.monthly_requests > 0 else 0.0
    return dict(inference=inference, notebooks=notebooks, training=training, pipelines=pipelines,
                storage=storage, logs=logs, total=total, unit=unit)

def vertex_cost(ss):
    endpoint = ss.vx_endpoint_nodes * ss.vx_endpoint_hours * ss.vx_endpoint_price_per_hour
    workbench = ss.vx_workbench_hours * ss.vx_workbench_price_per_hour
    training = ss.vx_train_hours * ss.vx_train_price_per_hour
    pipelines = ss.vx_pipe_hours * ss.vx_pipe_price_per_hour
    storage = ss.vx_storage_gb * ss.vx_gcs_price_per_gb
    logs = ss.logs_gb * ss.vx_logs_price_per_gb
    total = endpoint + workbench + training + pipelines + storage + logs
    unit_incl_idle = (endpoint / ss.monthly_requests) if ss.monthly_requests > 0 else 0.0
    return dict(endpoint=endpoint, workbench=workbench, training=training, pipelines=pipelines,
                storage=storage, logs=logs, total=total, unit=unit_incl_idle)

def databricks_cost(ss):
    notebooks = ss.dbx_nb_hours * (ss.dbx_nb_dbus_per_hour * ss.dbx_nb_price_per_dbu + ss.dbx_nb_ec2_per_hour)
    training = ss.dbx_job_hours * (ss.dbx_job_dbus_per_hour * ss.dbx_job_price_per_dbu + ss.dbx_job_ec2_per_hour)
    pipelines = ss.dbx_pipe_hours * (ss.dbx_pipe_dbus_per_hour * ss.dbx_pipe_price_per_dbu + ss.dbx_pipe_ec2_per_hour)
    serving = ss.dbx_serve_hours * (ss.dbx_serve_dbus_per_hour * ss.dbx_serve_price_per_dbu + ss.dbx_serve_ec2_per_hour)
    storage = ss.dbx_storage_gb * ss.dbx_s3_price_per_gb
    logs = ss.logs_gb * ss.dbx_logs_price_per_gb
    total = notebooks + training + pipelines + serving + storage + logs
    unit_serving = (serving / ss.monthly_requests) if ss.monthly_requests > 0 else 0.0
    return dict(notebooks=notebooks, training=training, pipelines=pipelines, serving=serving,
                storage=storage, logs=logs, total=total, unit=unit_serving)

# ---------- App ----------
ensure_defaults()

st.title("Stakeholder Cost Planner – DS Platforms")
st.caption("Interactive, stakeholder-friendly cost comparison for AWS SageMaker, Google Vertex AI, and Databricks. List-price style defaults – adjust for your org’s discounts.")

with st.sidebar:
    st.header("Quick Inputs")
    # Preset with apply
    preset = st.selectbox("Scenario preset", ["Lean MVP", "Pilot customer", "10× scale", "Custom"], index=0)
    if st.button("Apply preset"):
        st.session_state.update(apply_preset(preset))

    st.number_input("Monthly scoring requests", min_value=0, step=1000, key="monthly_requests")
    st.number_input("Seconds of compute / request", min_value=0.0, step=0.1, key="seconds_per_request")
    st.number_input("Serverless memory (GB)", min_value=0.5, step=0.5, key="serverless_mem_gb")
    st.number_input("Logs ingest per month (GB)", min_value=0, step=1, key="logs_gb")

    adv = st.checkbox("Show advanced platform knobs", value=False)

# Compute
sm = sagemaker_cost(st.session_state)
vx = vertex_cost(st.session_state)
dbx = databricks_cost(st.session_state)

# KPI row
c1, c2, c3 = st.columns(3)
totals = {"SageMaker": sm["total"], "Vertex AI": vx["total"], "Databricks": dbx["total"]}
cheapest = min(totals, key=totals.get)

with c1:
    st.metric("SageMaker – Total", fmt(sm["total"]), help="Includes serverless inference (no idle), notebooks, training, pipelines, storage & logs.")
    st.caption(f"Unit (/req, marginal): {fmt(sm['unit'])}")
    if cheapest == "SageMaker":
        st.success("Lowest total", icon="✅")
with c2:
    st.metric("Vertex AI – Total", fmt(vx["total"]), help="Includes endpoint idle baseline, workbench, training, pipelines, storage & logs.")
    st.caption(f"Unit (/req incl. idle): {fmt(vx['unit'])}")
    if cheapest == "Vertex AI":
        st.success("Lowest total", icon="✅")
with c3:
    st.metric("Databricks – Total", fmt(dbx["total"]), help="Includes notebooks, jobs, pipelines, (optional serving), storage & logs.")
    st.caption(f"Unit (/req if serving): {fmt(dbx['unit'])}")
    if cheapest == "Databricks":
        st.success("Lowest total", icon="✅")

# Chart
df_chart = pd.DataFrame([
    {"Platform": "SageMaker", "Total": sm["total"]},
    {"Platform": "Vertex AI", "Total": vx["total"]},
    {"Platform": "Databricks", "Total": dbx["total"]},
])
chart = (
    alt.Chart(df_chart).mark_bar().encode(
        x=alt.X("Platform:N", title=None),
        y=alt.Y("Total:Q", title="Monthly Total ($)", axis=alt.Axis(format="$,.0f")),
        tooltip=[alt.Tooltip("Platform:N"), alt.Tooltip("Total:Q", format="$,.2f")]
    ).properties(height=280)
)
st.altair_chart(chart, use_container_width=True)

# Breakdown tables
st.subheader("Breakdown by platform")
t1, t2, t3 = st.columns(3)
with t1:
    df_sm = pd.DataFrame({
        "Component": ["Inference", "Notebooks", "Training", "Pipelines", "Storage", "Logs", "Total"],
        "Cost ($)": [sm["inference"], sm["notebooks"], sm["training"], sm["pipelines"], sm["storage"], sm["logs"], sm["total"]]
    })
    st.dataframe(df_sm, use_container_width=True, height=260)
with t2:
    df_vx = pd.DataFrame({
        "Component": ["Endpoint baseline", "Workbench", "Training", "Pipelines", "Storage", "Logs", "Total"],
        "Cost ($)": [vx["endpoint"], vx["workbench"], vx["training"], vx["pipelines"], vx["storage"], vx["logs"], vx["total"]]
    })
    st.dataframe(df_vx, use_container_width=True, height=260)
with t3:
    df_dbx = pd.DataFrame({
        "Component": ["Notebooks", "Training", "Pipelines", "Serving", "Storage", "Logs", "Total"],
        "Cost ($)": [dbx["notebooks"], dbx["training"], dbx["pipelines"], dbx["serving"], dbx["storage"], dbx["logs"], dbx["total"]]
    })
    st.dataframe(df_dbx, use_container_width=True, height=260)

# CSV export
df_export = pd.DataFrame([
    {"Platform": "SageMaker", "Total": sm["total"], "Unit($/req)": sm["unit"], "Idle Floor ($/mo)": 0.0, "Notes": "Serverless scales to zero"},
    {"Platform": "Vertex AI", "Total": vx["total"], "Unit($/req)": vx["unit"], "Idle Floor ($/mo)": vx["endpoint"], "Notes": "Endpoint billed per node-hour (no scale-to-zero)"},
    {"Platform": "Databricks", "Total": dbx["total"], "Unit($/req)": dbx["unit"], "Idle Floor ($/mo)": None, "Notes": "Serving cost only if modeled via DBU/h"},
])
st.download_button("Download CSV snapshot", data=df_export.to_csv(index=False), file_name="ds-platform-costs.csv", mime="text/csv")

# Advanced knobs
if adv:
    st.markdown("---")
    st.header("Advanced platform knobs")

    # SageMaker
    with st.expander("AWS SageMaker – rates & hours", expanded=False):
        c = st.columns(5)
        with c[0]: st.number_input("Serverless $/sec/GB", min_value=0.0, step=0.000001, key="sm_rate_per_sec_gb")
        with c[1]: st.number_input("Notebook $/h", min_value=0.0, step=0.01, key="sm_notebook_price_per_hour")
        with c[2]: st.number_input("Training $/h", min_value=0.0, step=0.01, key="sm_train_price_per_hour")
        with c[3]: st.number_input("Pipelines $/h", min_value=0.0, step=0.01, key="sm_pipe_price_per_hour")
        with c[4]: st.number_input("S3 $/GB", min_value=0.0, step=0.001, key="sm_s3_price_per_gb")
        c = st.columns(5)
        with c[0]: st.number_input("Logs $/GB", min_value=0.0, step=0.01, key="sm_logs_price_per_gb")
        with c[1]: st.number_input("Notebook hours", min_value=0, step=1, key="sm_notebook_hours")
        with c[2]: st.number_input("Training hours", min_value=0, step=1, key="sm_train_hours")
        with c[3]: st.number_input("Pipelines hours", min_value=0, step=1, key="sm_pipe_hours")
        with c[4]: st.number_input("S3 storage (GB)", min_value=0, step=10, key="sm_storage_gb")
        st.caption("Unit cost = seconds × GB × $/sec/GB ÷ requests. No idle floor for serverless.")

    # Vertex
    with st.expander("Google Vertex AI – rates & hours", expanded=False):
        c = st.columns(5)
        with c[0]: st.number_input("Endpoint nodes", min_value=0, step=1, key="vx_endpoint_nodes")
        with c[1]: st.number_input("Endpoint hours/mo", min_value=0, step=1, key="vx_endpoint_hours")
        with c[2]: st.number_input("Endpoint $/h (per node)", min_value=0.0, step=0.01, key="vx_endpoint_price_per_hour")
        with c[3]: st.number_input("Workbench $/h", min_value=0.0, step=0.01, key="vx_workbench_price_per_hour")
        with c[4]: st.number_input("Training $/h", min_value=0.0, step=0.01, key="vx_train_price_per_hour")
        c = st.columns(5)
        with c[0]: st.number_input("Pipelines $/h", min_value=0.0, step=0.01, key="vx_pipe_price_per_hour")
        with c[1]: st.number_input("GCS $/GB", min_value=0.0, step=0.001, key="vx_gcs_price_per_gb")
        with c[2]: st.number_input("Logs $/GB", min_value=0.0, step=0.01, key="vx_logs_price_per_gb")
        with c[3]: st.number_input("Workbench hours", min_value=0, step=1, key="vx_workbench_hours")
        with c[4]: st.number_input("Training hours", min_value=0, step=1, key="vx_train_hours")
        c = st.columns(3)
        with c[0]: st.number_input("Pipelines hours", min_value=0, step=1, key="vx_pipe_hours")
        with c[1]: st.number_input("GCS storage (GB)", min_value=0, step=10, key="vx_storage_gb")
        st.caption("Unit (/req incl. idle) = endpoint baseline ÷ requests. Endpoints don't scale to zero.")

    # Databricks
    with st.expander("Databricks – rates & hours", expanded=False):
        c = st.columns(5)
        with c[0]: st.number_input("NB DBUs/h", min_value=0.0, step=0.1, key="dbx_nb_dbus_per_hour")
        with c[1]: st.number_input("NB $/DBU", min_value=0.0, step=0.01, key="dbx_nb_price_per_dbu")
        with c[2]: st.number_input("NB EC2 $/h", min_value=0.0, step=0.001, key="dbx_nb_ec2_per_hour")
        with c[3]: st.number_input("NB hours", min_value=0, step=1, key="dbx_nb_hours")
        with c[4]: st.number_input("Jobs DBUs/h", min_value=0.0, step=0.1, key="dbx_job_dbus_per_hour")
        c = st.columns(5)
        with c[0]: st.number_input("Jobs $/DBU", min_value=0.0, step=0.01, key="dbx_job_price_per_dbu")
        with c[1]: st.number_input("Jobs EC2 $/h", min_value=0.0, step=0.001, key="dbx_job_ec2_per_hour")
        with c[2]: st.number_input("Jobs hours", min_value=0, step=1, key="dbx_job_hours")
        with c[3]: st.number_input("Pipes DBUs/h", min_value=0.0, step=0.1, key="dbx_pipe_dbus_per_hour")
        with c[4]: st.number_input("Pipes $/DBU", min_value=0.0, step=0.01, key="dbx_pipe_price_per_dbu")
        c = st.columns(5)
        with c[0]: st.number_input("Pipes EC2 $/h", min_value=0.0, step=0.001, key="dbx_pipe_ec2_per_hour")
        with c[1]: st.number_input("Pipes hours", min_value=0, step=1, key="dbx_pipe_hours")
        with c[2]: st.number_input("Serving DBUs/h", min_value=0.0, step=0.1, key="dbx_serve_dbus_per_hour")
        with c[3]: st.number_input("Serving $/DBU", min_value=0.0, step=0.01, key="dbx_serve_price_per_dbu")
        with c[4]: st.number_input("Serving EC2 $/h", min_value=0.0, step=0.001, key="dbx_serve_ec2_per_hour")
        c = st.columns(3)
        with c[0]: st.number_input("Serving hours", min_value=0, step=1, key="dbx_serve_hours")
        with c[1]: st.number_input("S3 storage (GB)", min_value=0, step=10, key="dbx_storage_gb")
        with c[2]: st.number_input("S3 $/GB", min_value=0.0, step=0.001, key="dbx_s3_price_per_gb")
        st.number_input("Logs $/GB", min_value=0.0, step=0.01, key="dbx_logs_price_per_gb")
        st.caption("Totals = (DBUs×$/DBU + EC2$/h)×hours + storage + logs. If Serving hours = 0, per-request shows 0 (not modeled).")

st.markdown("---")
with st.expander("How to interpret & present", expanded=False):
    st.markdown("""
1. **Pick a preset** to match the scenario (Lean MVP, Pilot, 10× scale) and click **Apply preset**.
2. **Adjust just three inputs** in the sidebar (requests, seconds, memory) during live reviews.
3. **Read the KPI cards**: each platform shows total & unit $/req. Vertex's unit includes **idle floor**.
4. **Export CSV** to include a snapshot of numbers in slides or emails.
5. **Advanced knobs** let engineering tweak instance rates/hours or add serving on Databricks.
    """)
    st.caption("All figures are list-price style for illustration. Apply your org’s discounts/commits for final numbers.")
