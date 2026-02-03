import sqlite3
import pandas as pd
# Database setup
def init_db():
    conn = sqlite3.connect('wellness.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY,
        date TEXT,
        sleep_quality INTEGER,  -- 1-10 scale
        workout_intensity INTEGER,  -- 1-10 scale
        stress_level INTEGER  -- 1-10 scale
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY,
        date TEXT,
        recommendation TEXT,
        feedback INTEGER  -- 1-5 rating
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY,
        date TEXT,
        sleep_analysis TEXT,
        workout_load TEXT,
        stress_detection TEXT,
        safety_flags TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

# Function to save user data
def save_data(date, sleep, workout, stress):
    conn = sqlite3.connect('wellness.db')
    c = conn.cursor()
    c.execute("INSERT INTO user_data (date, sleep_quality, workout_intensity, stress_level) VALUES (?, ?, ?, ?)",
              (date, sleep, workout, stress))
    conn.commit()
    conn.close()

# Function to load data
def load_data():
    conn = sqlite3.connect('wellness.db')
    df = pd.read_sql_query("SELECT * FROM user_data ORDER BY date", conn)
    conn.close()
    return df

# Function to save recommendation and feedback
def save_recommendation(date, rec, feedback=None):
    conn = sqlite3.connect('wellness.db')
    c = conn.cursor()
    c.execute("INSERT INTO recommendations (date, recommendation, feedback) VALUES (?, ?, ?)",
              (date, rec, feedback))
    conn.commit()
    conn.close()

# Function to save analysis results
def save_analysis(date, sleep_analysis, workout_load, stress_detection, safety_flags):
    conn = sqlite3.connect('wellness.db')
    c = conn.cursor()
    c.execute("INSERT INTO analysis_results (date, sleep_analysis, workout_load, stress_detection, safety_flags) VALUES (?, ?, ?, ?, ?)",
              (date, sleep_analysis, workout_load, stress_detection, safety_flags))
    conn.commit()
    conn.close()

# Function to load analysis results
def load_analysis():
    conn = sqlite3.connect('wellness.db')
    df = pd.read_sql_query("SELECT * FROM analysis_results ORDER BY date DESC", conn)
    conn.close()
    return df
# Function to load recommendations
def load_recommendations():
    conn = sqlite3.connect('wellness.db')
    df = pd.read_sql_query("SELECT * FROM recommendations ORDER BY date DESC", conn)
    conn.close()
    return df
