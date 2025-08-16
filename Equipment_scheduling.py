import streamlit as st
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Job-Shop Equipment Optimization App")
st.subheader("By Adefavour")

st.markdown("""
This app solves the optimization problem for assigning jobs to machines in a job-shop to minimize total machine idle time (maximize utilization).
\nEnter the number of jobs and machines.
\nEdit the processing times table (positive values only; assume all jobs can run on all machines).
\nClick 'Optimize' to solve.
""")

# Input for number of jobs and machines
num_jobs = st.number_input("Number of jobs", min_value=1, value=3, step=1)
num_machines = st.number_input("Number of machines", min_value=1, value=2, step=1)

# Session state for processing times dataframe with reset confirmation
if 'processing_times' not in st.session_state or st.session_state.processing_times.shape != (num_jobs, num_machines):
    if 'processing_times' in st.session_state:
        if st.button("Confirm reset due to changed dimensions?"):
            initial_data = np.random.randint(1, 10, size=(num_jobs, num_machines))
            processing_times_df = pd.DataFrame(
                initial_data,
                columns=[f"Machine {m+1}" for m in range(num_machines)],
                index=[f"Job {i+1}" for i in range(num_jobs)]
            )
            st.session_state.processing_times = processing_times_df
    else:
        initial_data = np.random.randint(1, 10, size=(num_jobs, num_machines))
        processing_times_df = pd.DataFrame(
            initial_data,
            columns=[f"Machine {m+1}" for m in range(num_machines)],
            index=[f"Job {i+1}" for i in range(num_jobs)]
        )
        st.session_state.processing_times = processing_times_df
else:
    processing_times_df = st.session_state.processing_times

# Display and edit processing times
st.subheader("Processing Times (hours) - Edit Below")
edited_df = st.data_editor(processing_times_df, num_rows="dynamic", use_container_width=True)
st.session_state.processing_times = edited_df

# Validate dimensions after editing
if edited_df.shape != (num_jobs, num_machines):
    st.error("Edited table dimensions do not match specified jobs and machines. Please adjust.")
else:
    # Optimize button
    if st.button("Optimize"):
        tau = edited_df.to_numpy()

        # Validate processing times
        if np.any(tau <= 0):
            st.error("All processing times must be positive numbers.")
        else:
            if num_jobs * num_machines > 20:
                st.warning("Large problem size may slow optimization. Consider reducing jobs or machines.")
            
            # Big M value
            M = np.sum(np.max(tau, axis=1))

            # Create PuLP model
            model = pulp.LpProblem("JobShop_Optimization", pulp.LpMinimize)

            # Decision variables
            z = pulp.LpVariable.dicts("z", ((i, m) for i in range(num_jobs) for m in range(num_machines)), cat="Binary")
            phi = pulp.LpVariable.dicts("phi", (i for i in range(num_jobs)), lowBound=0, cat="Continuous")
            cmax = pulp.LpVariable("cmax", lowBound=0, cat="Continuous")

            # Sequencing variables
            seq = pulp.LpVariable.dicts(
                "seq",
                ((i, k, m) for i in range(num_jobs) for k in range(i + 1, num_jobs) for m in range(num_machines)),
                cat="Binary"
            )

            # Objective: Minimize total idle time
            total_busy = pulp.lpSum(tau[i, m] * z[(i, m)] for i in range(num_jobs) for m in range(num_machines))
            model += num_machines * cmax - total_busy

            # Constraints
            # 1. Each job assigned to exactly one machine
            for i in range(num_jobs):
                model += pulp.lpSum(z[(i, m)] for m in range(num_machines)) == 1

            # 2. Completion time of each job <= cmax
            for i in range(num_jobs):
                model += phi[i] + pulp.lpSum(tau[i, m] * z[(i, m)] for m in range(num_machines)) <= cmax

            # 3. No overlapping on the same machine (corrected sequencing)
            for i in range(num_jobs):
                for k in range(i + 1, num_jobs):
                    for m in range(num_machines):
                        model += phi[i] + tau[i, m] * z[(i, m)] <= phi[k] + M * (1 - seq[(i, k, m)] + (1 - z[(i, m)]) + (1 - z[(k, m)]))
                        model += phi[k] + tau[k, m] * z[(k, m)] <= phi[i] + M * (seq[(i, k, m)] + (1 - z[(i, m)]) + (1 - z[(k, m)]))

            # Solve the model
            solver_status = model.solve(pulp.PULP_CBC_CMD(msg=False))

            if solver_status == pulp.LpStatusOptimal:
                st.success("Optimal solution found!")
                assignments = {}
                start_times = {}
                completion_times = {}
                for i in range(num_jobs):
                    phi_val = pulp.value(phi[i])
                    start_times[i] = phi_val
                    assigned_m = None
                    proc_time = 0
                    for m in range(num_machines):
                        if pulp.value(z[(i, m)]) >= 0.5:
                            assigned_m = m
                            proc_time = tau[i, m]
                    assignments[i] = assigned_m
                    completion_times[i] = phi_val + proc_time

                makespan = pulp.value(cmax)
                busy_time = pulp.value(total_busy)
                idle_time = num_machines * makespan - busy_time
                utilization = (busy_time / (num_machines * makespan)) * 100 if makespan > 0 else 0

                st.subheader("Optimization Results")
                st.write(f"Makespan (C_max): {makespan:.2f} hours")
                st.write(f"Total busy time: {busy_time:.2f} hours")
                st.write(f"Total idle time: {idle_time:.2f} hours")
                st.write(f"Machine utilization: {utilization:.2f}%")

                st.subheader("Job Assignments and Schedules")
                results_df = pd.DataFrame({
                    "Job": [f"Job {i+1}" for i in range(num_jobs)],
                    "Assigned Machine": [f"Machine {assignments[i]+1}" for i in range(num_jobs)],
                    "Start Time": [f"{start_times[i]:.2f}" for i in range(num_jobs)],
                    "Completion Time": [f"{completion_times[i]:.2f}" for i in range(num_jobs)],
                    "Processing Time": [f"{tau[i, assignments[i]]:.2f}" for i in range(num_jobs)]
                })
                st.dataframe(results_df)

                st.subheader("Gantt Chart")
                fig, axs = plt.subplots(num_machines, 1, figsize=(12, 3 * num_machines), sharex=True)
                if num_machines == 1:
                    axs = [axs]
                colors = plt.cm.tab10(np.linspace(0, 1, num_jobs))

                for m in range(num_machines):
                    ax = axs[m]
                    ax.set_title(f"Machine {m+1}")
                    ax.set_ylabel("Job")
                    ax.set_xlabel("Time (hours)")
                    ax.grid(True)

                    jobs_on_m = [i for i in range(num_jobs) if assignments[i] == m]
                    if not jobs_on_m:
                        ax.text(0.5, 0.5, "No jobs assigned", ha='center', va='center', transform=ax.transAxes)
                    else:
                        for idx, i in enumerate(jobs_on_m):
                            start = start_times[i]
                            duration = tau[i, m]
                            ax.barh(idx, duration, left=start, height=0.8, color=colors[i], edgecolor='black')
                            ax.text(start + duration / 2, idx, f"Job {i+1}", ha='center', va='center', color='white')
                        ax.set_yticks(range(len(jobs_on_m)))
                        ax.set_yticklabels([f"Job {i+1}" for i in jobs_on_m])
                    ax.set_xlim(0, makespan + 1)

                plt.tight_layout()
                st.pyplot(fig)

            elif solver_status == pulp.LpStatusInfeasible:
                st.error("Problem is infeasible. Check constraints or inputs.")
            elif solver_status == pulp.LpStatusUnbounded:
                st.error("Problem is unbounded. Check objective or constraints.")
            else:
                st.error("No optimal solution found. Try adjusting inputs.")
