import streamlit as st
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Job-Shop Equipment Optimization App")
st.subheader("By Adefavour")
st.write("Last Updated: 10:56 PM WAT, Saturday, August 16, 2025")

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
        st
