import streamlit as st
import pickle
import pandas as pd

# Load the prediction model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Teams and cities for selection
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Page configuration
st.set_page_config(page_title="IPL Win Probability Calculator", page_icon="🏆", layout="wide")

# Add custom CSS for styling
st.markdown(
    """
    <style>
        /* Apply gradient background to the main app container */
        .stApp {
            background: linear-gradient(to bottom right, #8B4513, #D2691E, #F4A460);
            background-size: cover;
            color: white;
        }
        h1 {
            background-color: #ff4b4b;
            color: white;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            text-decoration: underline;
        }
        .stButton > button {
            background-color: #1e88e5;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
        }
        .stNumberInput > div > input, .stSelectbox > div {
            border-radius: 10px;
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }
        .stSelectbox label, .stNumberInput label {
            color: #f1c40f;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to bottom right, #8B4513, #D2691E, #F4A460);
            color: white;
        }
        h1 {
            background-color: #ff4b4b;
            color: white;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            text-decoration: underline;
        }
        .stButton > button {
            background-color: #1e88e5;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
        }
        .stNumberInput > div > input, .stSelectbox > div {
            border-radius: 10px;
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }
        .stSelectbox label, .stNumberInput label {
            color: #f1c40f;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title('IPL Win Probability Calculator')

# Input columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0, step=1)

# Score inputs
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, step=1)

with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1, format="%.1f")

with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Prediction button
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [remaining_wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    loss_prob = result[0][0]
    win_prob = result[0][1]

    st.subheader(f"🏏 {batting_team} - {round(win_prob * 100)}%")
    st.subheader(f"🏐 {bowling_team} - {round(loss_prob * 100)}%")
