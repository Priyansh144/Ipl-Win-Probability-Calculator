# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')
match
delivery

# %%
delivery.head()

# %%
#batting team #bowling team #city #runs left #balls left #wickets left #total runs #crr #rrr #result


# %%
total_score_df=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df

# %%
total_score_df=total_score_df[total_score_df['inning']==1]
total_score_df

# %%
match_df=match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')

# %%
match_df

# %%
match_df['team1'].unique()

# %%
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

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
match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]

# %%
match_df=match_df[match_df['dl_applied']==0]
match_df

# %%
match_df=match_df[['match_id','city','winner','total_runs']]
delivery_df=match_df.merge(delivery,on='match_id')

# %%
delivery_df=delivery_df[delivery_df['inning']==2]
#deliveries of all the matches of 2nd inning

# %%
#runs_left crr and rrr balls left wicketsleft are left
#runs_left=cumsum of total+runs_y


# %%

delivery_df['total_runs_y'] = pd.to_numeric(delivery_df['total_runs_y'], errors='coerce')
if delivery_df['total_runs_y'].isna().any():
    print("Found NaN values in 'total_runs_y'")
    delivery_df['total_runs_y'] = delivery_df['total_runs_y'].fillna(0)  # Replace NaN with 0 or appropriate value

delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
delivery_df

# %%
#runs left=totalruns-currentscore
delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score']
delivery_df[delivery_df['match_id']==1]

# %%
#balls left
delivery_df['balls_left']=126-(delivery_df['over']*6+delivery_df['ball'])
delivery_df

# %%
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")

# Step 2: Convert 'player_dismissed' to numeric (0 or 1)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 1 if x != "0" else 0)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype(int)

# Step 3: Calculate cumulative wickets for each match
delivery_df['wickets_fallen'] = delivery_df.groupby('match_id')['player_dismissed'].cumsum()

# Step 4: Calculate remaining wickets
delivery_df['wickets'] = 10 - delivery_df['wickets_fallen']
delivery_df

# %%
#crr=runs/overs
delivery_df['crr']=(delivery_df['current_score']*6)/(120-delivery_df['balls_left'])
delivery_df['rrr']=(delivery_df['runs_left']*6)/(delivery_df['balls_left'])
delivery_df

# %%
def result(row):
    return 1 if row['batting_team']==row['winner'] else 0

# %%
delivery_df['result'] = delivery_df.apply(result,axis=1)

# %%
final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]

# %%
final_df = final_df.sample(final_df.shape[0])
final_df.sample()
final_df = final_df[final_df['balls_left'] != 0]
final_df.dropna(inplace=True)

# %%
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# %%
X_train

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# %%
pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# %%
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# %%

y_pred = pipe.predict(X_test)

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# %%
pipe.predict_proba(X_test)[10]

# %%
import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


