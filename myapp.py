import streamlit as st
import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import google.generativeai as genai
import os
from datetime import datetime
import numpy as np  # For anomaly detection
from database import save_data,save_analysis,save_recommendation,load_analysis,load_data,load_recommendations
# Configure Gemini API (replace with your key)
genai.configure(api_key="AIzaSyCM-GRHF6x1YOZHurFxAm17iOcZTOJjSNo")  # Set this in your environment


# Recovery Analysis Engine Components
# Sleep Analysis
def analyze_sleep(df):
    if df.empty:
        return "No data available."
    avg_sleep = df['sleep_quality'].mean()
    sleep_trend = "Stable"
    if len(df) > 1:
        model = LinearRegression()
        df['date_num'] = pd.to_datetime(df['date']).map(pd.Timestamp.toordinal)
        model.fit(df[['date_num']], df['sleep_quality'])
        slope = model.coef_[0]
        sleep_trend = "Improving" if slope > 0 else "Declining" if slope < -0.1 else "Stable"
    
    analysis = f"Average sleep: {avg_sleep:.1f}/10.\n Trend: {sleep_trend}.\n "
    if avg_sleep < 5:
        analysis += "Poor sleep detected – prioritize recovery."
    return analysis

# Workout Load
def analyze_workout_load(df):
    if df.empty:
        return "No data available."
    rolling_avg = df['workout_intensity'].rolling(window=3).mean().iloc[-1] if len(df) >= 3 else df['workout_intensity'].mean()
    high_load_days = (df['workout_intensity'] > 8).sum()
    analysis = f"Rolling 3-day workout load: {rolling_avg:.1f}/10. High-intensity days: {high_load_days}."
    if high_load_days >= 3:
        analysis += " Potential overtraining – reduce intensity."
    return analysis

# Stress Detection
def detect_stress(df):
    if df.empty:
        return "No data available."
    avg_stress = df['stress_level'].mean()
    # Simple anomaly detection: z-score for spikes
    if len(df) > 1:
        mean_stress = df['stress_level'].mean()
        std_stress = df['stress_level'].std()
        recent_stress = df['stress_level'].iloc[-1]
        z_score = (recent_stress - mean_stress) / std_stress if std_stress > 0 else 0
        spike = "Spike detected" if z_score > 1.5 else "No spike"
    else:
        spike = "Not enough data"
    
    analysis = f"Average stress: {avg_stress:.1f}/10. Recent: {spike}."
    if avg_stress > 7:
        analysis += " High stress – recovery needed."
    return analysis

# Safety Rules
def apply_safety_rules(df, sleep, workout, stress):
    flags = []
    if stress > 8:
        flags.append("High stress (>8) – Recommend immediate rest only.")
    if sleep < 3:
        flags.append("Very poor sleep (<3) – Prioritize rest and light recovery.")
    if workout > 8 and len(df) >= 3 and df['workout_intensity'].tail(3).mean() > 8:
        flags.append("High workout load – Avoid intense activities; focus on rest.")
    if not flags:
        flags.append("No safety flags – Proceed with recommendations.")
    return "; ".join(flags)


# Simple ML: Detect trends (e.g., increasing stress)
def analyze_trends(df):
    if len(df) < 2:
        return "Not enough data for trend analysis."
    
    # Linear regression on stress over time
    df['date_num'] = pd.to_datetime(df['date']).map(pd.Timestamp.toordinal)
    model = LinearRegression()
    model.fit(df[['date_num']], df['stress_level'])
    slope = model.coef_[0]
    
    if slope > 0.1:
        return "Trend: Stress levels are increasing. Recovery needed."
    elif df['sleep_quality'].mean() < 5:
        return "Trend: Poor sleep quality detected. Suggest rest."
    else:
        return "Trends look stable. Maintain habits."

# AI Recommendation using Gemini (now includes safety flags)
def generate_recommendation(trend, sleep, workout, stress, safety_flags):
    prompt = f"""
    Based on user data: Sleep quality {sleep}/10, Workout intensity {workout}/10, Stress level {stress}/10.
    Trend analysis: {trend}.
    Safety flags: {safety_flags}.
    Generate personalized, safe recovery recommendations for general wellness (not medical advice).
    Include: 1 rest day suggestion, 1 stretching routine, 1 light activity idea.
    Adjust for safety (e.g., if high risk, suggest rest only). Keep it ethical, preventive, and recovery-focused. Limit to 200 words.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
    except:
        return "Unable to connect to LLM"
    return response.text

# Mapping functions for user-friendly inputs
def map_sleep_to_num(choice):
    mapping = {"Very Poor": 1, "Poor": 3, "Fair": 5, "Good": 7, "Excellent": 10}
    return mapping.get(choice, 5)

def map_workout_to_num(choice):
    mapping = {"Very Light": 1, "Light": 3, "Moderate": 5, "Intense": 7, "Very Intense": 10}
    return mapping.get(choice, 5)

def map_stress_to_num(choice):
    mapping = {"Very Low": 1, "Low": 3, "Moderate": 5, "High": 7, "Very High": 10}
    return mapping.get(choice, 5)

# Streamlit UI
st.title("Personalized Recovery & Rest Recommendation System Using AI")
st.markdown("This app provides general wellness guidance for preventive recovery. It is not medical advice.")

# Sidebar for data input
st.sidebar.header("Share How You Felt Today")
st.sidebar.markdown("Rate based on your overall experience. Hover for examples!")

date = st.sidebar.date_input("Date", datetime.today())

sleep_options = ["Very Poor", "Poor", "Fair", "Good", "Excellent"]
sleep_choice = st.sidebar.selectbox(
    "How was your sleep last night?",
    sleep_options,
    help="Very Poor: I felt exhausted and unrested all day. Excellent: I woke up refreshed and energized."
)
sleep = map_sleep_to_num(sleep_choice)

workout_options = ["Very Light", "Light", "Moderate", "Intense", "Very Intense"]
workout_choice = st.sidebar.selectbox(
    "How intense was your workout today?",
    workout_options,
    help="Very Light: A short walk. Very Intense: Heavy lifting or long run that left me very tired."
)
workout = map_workout_to_num(workout_choice)

stress_options = ["Very Low", "Low", "Moderate", "High", "Very High"]
stress_choice = st.sidebar.selectbox(
    "How stressed did you feel today?",
    stress_options,
    help="Very Low: Calm and relaxed. Very High: Overwhelmed and anxious most of the day."
)
stress = map_stress_to_num(stress_choice)

if st.sidebar.button("Save My Feelings"):
    save_data(str(date), sleep, workout, stress)
    st.sidebar.success("Thanks! Your data is saved.")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Data & Trends", "Recovery Analysis", "Recommendations", "History & Feedback"])

with tab1:
    st.header("Your Data & Trend Analysis")
    df = load_data()
    if not df.empty:
        st.dataframe(df)
        # Visualization
        fig = px.line(df, x='date', y=['sleep_quality', 'workout_intensity', 'stress_level'], title="Trends Over Time")
        st.plotly_chart(fig)
        # Analysis
        trend = analyze_trends(df)
        st.subheader("Trend Analysis")
        st.write(trend)
    else:
        st.write("No data yet. Add some in the sidebar!")

with tab2:
    st.header("Recovery Analysis Engine")
    df = load_data()
    if not df.empty:
        # Run analyses
        sleep_analysis = analyze_sleep(df)
        workout_load = analyze_workout_load(df)
        stress_detection = detect_stress(df)
        safety_flags = apply_safety_rules(df, sleep, workout, stress)
        
        # Display results
        st.subheader("Sleep Analysis")
        st.text(sleep_analysis)
        st.subheader("Workout Load")
        st.write(workout_load)
        st.subheader("Stress Detection")
        st.write(stress_detection)
        st.subheader("Safety Rules & Flags")
        st.write(safety_flags)
        
        # Save analysis
        save_analysis(str(datetime.today()), sleep_analysis, workout_load, stress_detection, safety_flags)
        st.success("Analysis saved!")
        
        # New Graphs for Recovery Analysis
        st.subheader("Visual Recovery Insights")
        
        # Graph 1: Sleep Quality Trends (Bar Chart)
        recent_df = df.tail(7)  # Last 7 days or available
        fig_sleep = px.bar(recent_df, x='date', y='sleep_quality', 
                           title="Sleep Quality Trends (Last 7 Days)",
                           labels={'sleep_quality': 'Sleep Quality (1-10)', 'date': 'Date'},
                           color='sleep_quality', color_continuous_scale=['red', 'yellow', 'green'])
        fig_sleep.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig_sleep)
        st.markdown("*This bar chart shows your sleep quality over time. Green means better sleep—aim for consistency to feel more rested!*")
        
        # Graph 2: Workout Load Over Time (Line Chart with Shading)
        df['rolling_workout'] = df['workout_intensity'].rolling(window=3).mean()
        fig_workout = px.line(df, x='date', y='rolling_workout', 
                              title="Workout Load (3-Day Rolling Average)",
                              labels={'rolling_workout': 'Workout Intensity (1-10)', 'date': 'Date'})
        # Add shaded zone for high load
        fig_workout.add_hrect(y0=8, y1=10, line_width=0, fillcolor="red", opacity=0.2, annotation_text="High Load Zone")
        st.plotly_chart(fig_workout)
        st.markdown("*This line shows your workout intensity trend. The red zone highlights potential overtraining—listen to your body and rest if needed!*")
        
        # Graph 3: Stress Levels with Anomalies (Scatter/Line Combo)
        df['stress_z'] = (df['stress_level'] - df['stress_level'].mean()) / df['stress_level'].std()
        fig_stress = px.scatter(df, x='date', y='stress_level', 
                                title="Stress Levels & Anomalies",
                                labels={'stress_level': 'Stress Level (1-10)', 'date': 'Date'},
                                color=df['stress_z'].abs() > 1.5, color_discrete_map={True: 'red', False: 'blue'})
        fig_stress.add_traces(px.line(df, x='date', y='stress_level').data)  # Add line for trend
        st.plotly_chart(fig_stress)
        st.markdown("*Blue dots are normal stress; red dots are spikes. Watch for patterns to spot early fatigue and prioritize recovery!*")
        
        # Graph 4: Safety Flags Summary (Pie Chart from History)
        analyses_df = load_analysis()
        if not analyses_df.empty:
            flag_counts = {}
            for flags in analyses_df['safety_flags']:
                for flag in flags.split('; '):
                    if flag not in ["No safety flags – Proceed with recommendations."]:
                        flag_counts[flag] = flag_counts.get(flag, 0) + 1
            if flag_counts:
                fig_flags = px.pie(names=list(flag_counts.keys()), values=list(flag_counts.values()), 
                                   title="Safety Flags Frequency (From Your History)")
                st.plotly_chart(fig_flags)
                st.markdown("*This pie chart shows how often safety concerns (like high stress) have been flagged. Use it to track your recovery habits!*")
            else:
                st.write("No safety flags in history yet.")
        
        # Graph 5: Overall Recovery Dashboard (Subplots)
        fig_dashboard = px.scatter(df, x='sleep_quality', y='stress_level', 
                                  title="Sleep vs. Stress Correlation",
                                  labels={'sleep_quality': 'Sleep Quality (1-10)', 'stress_level': 'Stress Level (1-10)'},
                                  trendline="ols")  # Add trendline for correlation
        st.plotly_chart(fig_dashboard)
        st.markdown("*This scatter plot shows how sleep and stress relate. A downward trend means better sleep might reduce stress—focus on rest for better recovery!*")
        
    else:
        st.write("Add data first to run analysis.")
with tab3:
    st.header("AI-Powered Recommendations")
    df = load_data()
    if not df.empty:
        trend = analyze_trends(df)
        safety_flags = apply_safety_rules(df, sleep, workout, stress)
        rec = generate_recommendation(trend, sleep, workout, stress, safety_flags)
        st.write(rec)
        # Save recommendation
        save_recommendation(str(datetime.today()), rec)
        # Feedback
        feedback = st.slider("Rate this recommendation (1-5)", 1, 5, 3)
        if st.button("Submit Feedback"):
            # Update last recommendation with feedback (simple approach)
            recs = load_recommendations()
            if not recs.empty:
                last_id = recs.iloc[0]['id']
                conn = sqlite3.connect('wellness.db')
                c = conn.cursor()
                c.execute("UPDATE recommendations SET feedback = ? WHERE id = ?", (feedback, last_id))
                conn.commit()
                conn.close()
            st.success("Feedback saved! Future recommendations will adapt.")
    else:
        st.write("Add data first to get recommendations.")

with tab4:
    st.header("History & Feedback")
    recs = load_recommendations()
    analyses = load_analysis()
    if not recs.empty:
        st.subheader("Recommendation History")
        st.dataframe(recs)
        avg_feedback = recs['feedback'].dropna().mean()
        st.write(f"Average feedback rating: {avg_feedback:.1f}/5")
    if not analyses.empty:
        st.subheader("Analysis History")
        st.dataframe(analyses)
    if recs.empty and analyses.empty:
        st.write("No history yet.")