import streamlit as st
import matplotlib.pyplot as plt
import json
from helper import *
st.title("RL course recommender simulation")

courses = load_courses()
visualize_courses(courses)

num_students = st.slider("Number of students", 5, 100, 10)
episodes = st.slider("Number of RL episodes", 10, 500, 100)

if st.button("Run Simulation"):
    students = generate_students(num_students, courses)
    st.success(f"Generated {num_students} students.")
    gpa_history, Q = RL(students, courses, episodes=episodes)
    st.success(f"Simulated {num_students} students for {episodes} episodes.")
    plot_gpa(gpa_history)

