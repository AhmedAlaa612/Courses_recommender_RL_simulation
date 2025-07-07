import networkx as nx
import random
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
def get_gpa(grades):
    """Calculate GPA from a list of grades."""
    if not grades:
        return 0.0
    total_points = sum(grade * 3 for grade in grades)  # all grades are 3-credit hours
    return total_points / (3 * len(grades))

def get_open_courses(courses, student_courses):
    """Returns a list of open courses that a student can take."""
    open_courses = []
    for course_id in courses.nodes():
        if course_id in student_courses:
            continue 
        prerequisites = list(courses.predecessors(course_id))
        if all(prereq in student_courses for prereq in prerequisites):
            open_courses.append(course_id)
            
    return open_courses

def generate_students(n, courses):
    """Generate a list of students with random courses and GPA."""
    students = []
    interests = ['Data', 'AI', 'Software']
    for i in range(n):
        semester = random.randint(1, 7)
        min_courses, max_courses = (3*semester, 5*semester)
        num_courses = random.randint(min_courses, max_courses)
        passed = []
        grades = []
        for _ in range(num_courses):
            c = random.choice(get_open_courses(courses, passed))
            grade = int(np.random.choice([0,1,2,3,4], p=[0.1,0.2,0.2,0.3,0.2])) # F, D, C, B, A
            if grade != 0:
                passed.append(c)
            grades.append({'id': c, 'grade': grade})
        gpa = float(get_gpa([grade['grade'] for grade in grades]))
        students.append({
            'id': f'{i+1}',
            'semester': semester,
            'courses': passed,
            'grades': grades,
            'gpa': gpa,
            'interests': random.sample(interests, k=random.randint(1, len(interests)))
        })
    return students

def get_actions(open_courses, max_comb=100):
    actions = []
    for courses_cnt in range(3, 6):
        actions.extend(itertools.combinations(open_courses, courses_cnt))
    if len(actions) > max_comb:
        actions = random.sample(actions, max_comb)
    return actions

def take_courses(student, action):
    """Simulate taking courses and return the new GPA."""
    student_courses = student['courses']
    student_grades = student['grades']
    for course in action:
        grade = random.choice([0, 1, 2, 3, 4])
        student_grades.append({'id': course, 'grade': grade})
        if grade > 0:
            student_courses.append(course)
    
    new_gpa = get_gpa([grade['grade'] for grade in student_grades])
    
    return new_gpa, student_courses, student_grades

def RL(students, courses, alpha=0.1, gamma=0.9, episodes=100):
    Q = {}
    gpa_history = {s['id'] : [round(s['gpa'], 1)] for s in students}
    for episode in range(episodes):
        for student in students:
            state = (
                tuple(sorted(student['courses'])),
                round(student['gpa'], 1),
                student['semester']
            )
            open_courses = get_open_courses(courses, student['courses'])
            if len(open_courses) < 3: continue

            possible_actions = get_actions(open_courses)
            if not possible_actions : continue

            # explore or exploit

            if state not in Q:
                Q[state] = {action: 0 for action in possible_actions}   
                action = random.choice(possible_actions)
            else:
                if random.random() < 0.2:
                    action = random.choice(possible_actions)
                else:
                    action = max(Q[state], key=Q[state].get)
            # take action
            new_gpa, new_courses, new_grades = take_courses(student, action)
            # reward
            reward = new_gpa - student['gpa']

            # new state 
            new_state = (
                tuple(sorted(new_courses)),
                round(new_gpa, 1),
                student['semester'] + 1
            )
            # update Q-value
            max_new_q = max(Q.get(new_state, {}).values(), default=0)
            Q[state][action] += alpha * (reward + gamma * max_new_q - Q[state][action])
            # update student info
            student['gpa'] = new_gpa
            student['courses'] = new_courses
            student['grades'] = new_grades
            student['semester'] += 1
            gpa_history[student['id']].append(round(new_gpa, 1))
    return gpa_history, Q


def load_courses():
    return nx.read_graphml("courses.graphml")

def plot_gpa(gpa_history):
    st.header("GPA Progress")
    fig, ax = plt.subplots(figsize=(10,6))
    for student_id, history in gpa_history.items():
        if int(student_id) <= 5:
            ax.plot(history, label=f"Student {student_id}")
    ax.set_xlabel("Semester")
    ax.set_ylabel("GPA")
    ax.set_title("GPA Progression")
    ax.legend()
    st.pyplot(fig)

def visualize_courses(courses):
    net = Network(height="400px", width="100%",directed=True)
    
    net.from_nx(courses)
    
    # Generate HTML as string
    html = net.generate_html()
    
    # Display in Streamlit
    components.html(html, height=400, width=900)
    