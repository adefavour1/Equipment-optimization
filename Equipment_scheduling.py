# scheduler_app.py
# Streamlit Job-Shop Scheduling App (OR-Tools) — red & grey theme
# Features:
# - Manual input via editable table or CSV upload
# - Solves job-shop scheduling with OR-Tools CP-SAT (minimize makespan)
# - Red & grey Gantt chart in-app
# - Export schedule + KPIs to CSV/Excel (editable, no image export)

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from ortools.sat.python import cp_model

# -------------------------
# Styling (Red & Grey)
# -------------------------
PRIMARY_RED = "#D32F2F"
PRIMARY_GREY = "#9E9E9E"
BG_GREY = "#f5f5f5"
TEXT_DARK = "#1a1a1a"

st.set_page_config(page_title="Job-Shop Scheduler (OR-Tools)", layout="wide")

st.markdown(f"""
<style>
    .stApp {{
        background: {BG_GREY};
        color: {TEXT_DARK};
    }}
    .app-title {{
        text-align:center;
        font-weight: 800;
        font-size: 32px;
        color: {PRIMARY_RED};
        margin-bottom: 0.2rem;
    }}
    .app-subtitle {{
        text-align:center;
        color: {PRIMARY_GREY};
        font-weight: 600;
        margin-bottom: 1rem;
    }}
    hr {{
        border: none;
        border-top: 2px solid {PRIMARY_GREY}33;
        margin: .25rem 0 1rem 0;
    }}
    .small-note {{
        color: #555;
        font-size: 0.9rem;
    }}
    .metric-header > div[data-testid="stMetricValue"] {{
        color: {PRIMARY_RED} !important;
    }}
</style>
<div class="app-title">Job-Shop Scheduling Optimizer</div>
<div class="app-subtitle">OR-Tools · Red/Grey Theme · Gantt · CSV/Excel Export</div>
<hr/>
""", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
def validate_df(df: pd.DataFrame) -> tuple[bool, str]:
    required_cols = ["job_id", "task_id", "machine", "duration"]
    for c in required_cols:
        if c not in df.columns:
            return False, f"Missing required column: `{c}`"
    if df.empty:
        return False, "Input table is empty."
    if (df["duration"] <= 0).any():
        return False, "All durations must be positive."
    # task_id should be integers/non-negative
    try:
        if (df["task_id"].astype(int) < 0).any():
            return False, "task_id must be >= 0."
    except Exception:
        return False, "task_id must be integers."
    # Sort consistency: we’ll sort inside solver, but warn if any gaps
    return True, ""

def build_and_solve_schedule(df: pd.DataFrame, time_limit_s: int | None = None):
    """
    Classic job-shop:
      - Each row is an operation (job_id, task_id, machine, duration)
      - Precedence: operations with the same job_id must follow task_id order
      - Machines cannot overlap
      - Minimize makespan
    Returns (status, schedule_df, makespan, solver_log)
    """
    # Normalize types
    df = df.copy()
    df["task_id"] = df["task_id"].astype(int)
    df["duration"] = df["duration"].astype(int)

    # Sort operations by job, then task
    df = df.sort_values(by=["job_id", "task_id"]).reset_index(drop=True)

    # Build index for quick lookup
    jobs = df["job_id"].unique().tolist()
    machines = sorted(df["machine"].astype(str).unique().tolist())

    # Map (row) -> (job, task, machine, dur)
    ops = []
    for idx, row in df.iterrows():
        ops.append((row["job_id"], row["task_id"], str(row["machine"]), int(row["duration"])))

    # OR-Tools model
    model = cp_model.CpModel()

    # Create variables
    horizon = int(df["duration"].sum())  # simple upper bound
    start_vars = {}
    end_vars = {}
    interval_vars = {}

    # Per-machine intervals to add NoOverlap
    machine_to_intervals = {m: [] for m in machines}

    for i, (job, task, mach, dur) in enumerate(ops):
        start = model.NewIntVar(0, horizon, f"start_{i}")
        end = model.NewIntVar(0, horizon, f"end_{i}")
        interval = model.NewIntervalVar(start, dur, end, f"interval_{i}")
        start_vars[i] = start
        end_vars[i] = end
        interval_vars[i] = interval
        machine_to_intervals[mach].append(interval)

    # Job precedence constraints
    # For each job, order tasks by task_id
    for job in jobs:
        job_rows = [(i, t) for i, (j, t, m, d) in enumerate(ops) if j == job]
        job_rows.sort(key=lambda x: x[1])  # sort by task_id
        for (i_prev, _), (i_next, __) in zip(job_rows, job_rows[1:]):
            model.Add(start_vars[i_next] >= end_vars[i_prev])

    # No overlap per machine
    for m in machines:
        model.AddNoOverlap(machine_to_intervals[m])

    # Makespan
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [end_vars[i] for i in end_vars])

    # Objective
    model.Minimize(makespan)

    # Solve
    solver = cp_model.CpSolver()
    if time_limit_s:
        solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    # Build outputs
    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status_str = status_map.get(status, str(status))

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        rows = []
        for i, (job, task, mach, dur) in enumerate(ops):
            s = int(solver.Value(start_vars[i]))
            e = int(solver.Value(end_vars[i]))
            rows.append({
                "job_id": job,
                "task_id": task,
                "machine": mach,
                "duration": dur,
                "start": s,
                "end": e,
            })
        sched = pd.DataFrame(rows).sort_values(["machine", "start", "job_id", "task_id"])
        return status_str, sched, int(solver.Value(makespan)), solver.ResponseStats()
    else:
        return status_str, pd.DataFrame(), None, solver.ResponseStats()

def gantt_figure(schedule_df: pd.DataFrame, color_map=None):
    """
    Build a Plotly Gantt from integer time units.
    We convert start/end to datetimes for nice axes.
    """
    if schedule_df.empty:
        return None

    # Base origin
    origin = datetime(2024, 1, 1, 8, 0, 0)
    df_plot = schedule_df.copy()
    df_plot["Start"] = df_plot["start"].apply(lambda x: origin + timedelta(minutes=int(x)))
    df_plot["Finish"] = df_plot["end"].apply(lambda x: origin + timedelta(minutes(int(x))))
    df_plot["Task"] = df_plot.apply(lambda r: f"Job {r['job_id']} — T{r['task_id']} ({r['machine']})", axis=1)

    # Assign red/grey colors by job
    jobs = df_plot["job_id"].astype(str).unique().tolist()
    default_map = {}
    for idx, j in enumerate(jobs):
        default_map[j] = PRIMARY_RED if idx % 2 == 0 else PRIMARY_GREY
    color_map = color_map or default_map

    fig = px.timeline(
        df_plot,
        x_start="Start",
        x_end="Finish",
        y="machine",
        color=df_plot["job_id"].astype(str),
        color_discrete_map=color_map,
        hover_data=["job_id", "task_id", "machine", "duration", "start", "end"],
        title="Gantt Chart (minutes)"
    )
    fig.update_yaxes(title="Machine")
    fig.update_xaxes(title="Time")
    fig.update_layout(
        legend_title_text="Job",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        bargap=0.2,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

def export_schedule_to_excel(inputs_df: pd.DataFrame, schedule_df: pd.DataFrame, makespan: int, filename: str):
    """
    Exports three sheets: Inputs, Schedule, KPIs (editable, no images).
    """
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        inputs_df.to_excel(writer, sheet_name="Inputs", index=False)
        schedule_df.to_excel(writer, sheet_name="Schedule", index=False)
        kpi_df = pd.DataFrame(
            [{"Metric": "Makespan (minutes)", "Value": makespan},
             {"Metric": "Jobs", "Value": inputs_df['job_id'].nunique()},
             {"Metric": "Machines", "Value": inputs_df['machine'].nunique()},
             {"Metric": "Operations", "Value": len(inputs_df)}]
        )
        kpi_df.to_excel(writer, sheet_name="KPIs", index=False)

# -------------------------
# Sidebar (options)
# -------------------------
with st.sidebar:
    st.markdown("### Options")
    mode = st.radio("Input Mode", ["Manual table", "Upload CSV"], index=0)
    st.caption("CSV columns required: job_id, task_id, machine, duration")
    time_limit = st.number_input("Solver time limit (seconds, optional)", min_value=0, value=0, step=1)
    run_limit = int(time_limit) if time_limit > 0 else None

# -------------------------
# Input Area
# -------------------------
st.markdown("### 1) Provide Job & Task Data")

if mode == "Upload CSV":
    f = st.file_uploader("Upload CSV with columns: job_id, task_id, machine, duration", type=["csv"])
    if f is not None:
        try:
            input_df = pd.read_csv(f)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            input_df = pd.DataFrame()
    else:
        input_df = pd.DataFrame()
else:
    # A small editable starter table
    starter = pd.DataFrame([
        {"job_id": "A", "task_id": 0, "machine": "M1", "duration": 30},
        {"job_id": "A", "task_id": 1, "machine": "M2", "duration": 20},
        {"job_id": "B", "task_id": 0, "machine": "M2", "duration": 25},
        {"job_id": "B", "task_id": 1, "machine": "M1", "duration": 15},
    ])
    st.caption("Edit cells directly. Add/delete rows with the + / trash icons.")
    input_df = st.data_editor(
        starter,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor_inputs"
    )

ok, msg = validate_df(input_df) if not input_df.empty else (False, "Provide input data.")

if not ok:
    st.warning(msg)
    st.stop()

# -------------------------
# Solve
# -------------------------
st.markdown("### 2) Solve the Schedule")
solve = st.button("🚀 Solve with OR-Tools", type="primary")

if not solve:
    st.stop()

status, schedule_df, makespan, stats = build_and_solve_schedule(input_df, time_limit_s=run_limit)

if status in ("OPTIMAL", "FEASIBLE"):
    st.success(f"Solver status: {status}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Makespan (minutes)", f"{makespan}")
    c2.metric("Jobs", f"{input_df['job_id'].nunique()}")
    c3.metric("Machines", f"{input_df['machine'].nunique()}")
    c4.metric("Operations", f"{len(input_df)}")

    st.markdown("### 3) Schedule (Table)")
    st.dataframe(schedule_df, use_container_width=True)

    st.markdown("### 4) Gantt Chart")
    fig = gantt_figure(schedule_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No schedule to plot.")

    # -------------------------
    # Exports
    # -------------------------
    st.markdown("### 5) Export Results")
    # CSV
    csv_bytes = schedule_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Schedule (CSV)",
        data=csv_bytes,
        file_name="schedule.csv",
        mime="text/csv"
    )

    # Excel (Inputs + Schedule + KPIs)
    excel_name = f"jobshop_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    try:
        export_schedule_to_excel(input_df, schedule_df, makespan, excel_name)
        with open(excel_name, "rb") as f:
            st.download_button(
                "⬇️ Download Full Report (Excel)",
                data=f.read(),
                file_name=excel_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except Exception as e:
        st.error(f"Excel export failed: {e}")

    with st.expander("Solver details (log)"):
        st.code(stats or "No stats", language="text")

else:
    st.error(f"Solver status: {status}. No feasible schedule was found.")
    with st.expander("Solver details (log)"):
        st.code(stats or "No stats", language="text")
