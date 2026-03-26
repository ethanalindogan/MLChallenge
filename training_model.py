
"""
train_model.py
==============
Run this script to train the logistic regression model on the dataset
and print out the weights and intercept.

Usage:
    python3 train_model.py

Make sure ml_challenge_dataset.csv is in the same folder.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ── Step 1: Load the data ──────────────────────────────────────────────────
df = pd.read_csv('ml_challenge_dataset.csv')
print(f"Loaded {len(df)} rows.")

# ── Step 2: Build features (X) ────────────────────────────────────────────
likert_map = {
    '1 - Strongly disagree': 1,
    '2 - Disagree':          2,
    '3 - Neutral/Unsure':    3,
    '4 - Agree':             4,
    '5 - Strongly agree':    5
}

emotion_cols = [
    'This art piece makes me feel sombre.',
    'This art piece makes me feel content.',
    'This art piece makes me feel calm.',
    'This art piece makes me feel uneasy.',
]
for col in emotion_cols:
    df[col+'_num'] = df[col].map(likert_map).fillna(3.0)

df['emotion_intensity'] = df['On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?'].fillna(6.0)
df['num_colours'] = df['How many prominent colours do you notice in this painting?'].clip(upper=20).fillna(3.0)
df['num_objects']  = df['How many objects caught your eye in the painting?'].clip(upper=20).fillna(3.0)

season_col = 'What season does this art piece remind you of?'
for s in ['Spring', 'Summer', 'Fall', 'Winter']:
    df['season_'+s.lower()] = df[season_col].fillna('').str.contains(s).astype(float)
df['season_missing'] = df[season_col].isna().astype(float)

df['food_text'] = df['If this painting was a food, what would be?'].fillna('').str.lower()
df['food_wl']  = df['food_text'].apply(lambda x: float(any(k in x for k in ['salad','watermelon','green','strawberry','cucumber','apple','fruit','vegetable','melon'])))
df['food_sn']  = df['food_text'].apply(lambda x: float(any(k in x for k in ['ice cream','blueberry','pasta','spaghetti','steak','cake','sushi','blue'])))
df['food_pom'] = df['food_text'].apply(lambda x: float(any(k in x for k in ['pizza','bread','cheese','soup','banana','noodle','stale','lasagna','fries'])))

df['room_text'] = df['If you could purchase this painting, which room would you put that painting in?'].fillna('').str.lower()
for r in ['office', 'bedroom', 'bathroom', 'living']:
    df['room_'+r] = df['room_text'].str.contains(r).astype(float)

df['desc_text'] = df['Describe how this painting makes you feel.'].fillna('').str.lower()
df['desc_melancholy'] = df['desc_text'].apply(lambda x: float(any(k in x for k in ['time','melt','clock','surreal','dread','anxious','dream','eerie','strange'])))
df['desc_peaceful']   = df['desc_text'].apply(lambda x: float(any(k in x for k in ['peace','calm','serene','nature','relax','tranquil','garden','water','pond','lily'])))
df['desc_awe']        = df['desc_text'].apply(lambda x: float(any(k in x for k in ['wonder','awe','inspire','sky','star','night','vast','infinite','beautiful'])))

feature_cols = (
    [c+'_num' for c in emotion_cols] +
    ['emotion_intensity', 'num_colours', 'num_objects'] +
    ['season_spring', 'season_summer', 'season_fall', 'season_winter', 'season_missing'] +
    ['food_wl', 'food_sn', 'food_pom'] +
    ['room_office', 'room_bedroom', 'room_bathroom', 'room_living'] +
    ['desc_melancholy', 'desc_peaceful', 'desc_awe']
)

X = df[feature_cols].values
print(f"Feature matrix shape: {X.shape}  (rows=surveys, cols=features)")

# ── Step 3: Build labels (y) ──────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(df['Painting'].values)
print(f"Classes (in order): {le.classes_}")
print(f"  0 = {le.classes_[0]}")
print(f"  1 = {le.classes_[1]}")
print(f"  2 = {le.classes_[2]}")

# ── Step 4: Train the model ────────────────────────────────────────────────
print("\nTraining logistic regression (C=5.0)...")
clf = LogisticRegression(C=5.0, max_iter=1000, solver='lbfgs')
clf.fit(X, y)

train_acc = np.mean(clf.predict(X) == y)
print(f"Training accuracy: {train_acc:.4f}")

# ── Step 5: Print the weights and intercept ───────────────────────────────
print("\n" + "="*60)
print("Copy-paste the lines below into pred.py")
print("="*60)
print()
print("WEIGHTS =", repr(clf.coef_.tolist()))
print()
print("INTERCEPT =", repr(clf.intercept_.tolist()))
print()
print("="*60)