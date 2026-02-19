# analysis.py

import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import linregress

def analyze(db_path):

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM steps", conn)

    print("Mean energy:", df.energy.mean())

    slopes = []
    for pid in df.problem_id.unique():
        sub = df[df.problem_id == pid]
        if len(sub) > 2:
            slope, *_ = linregress(sub.iteration, sub.energy)
            slopes.append(slope)

    print("Mean slope:", np.mean(slopes))

    # Intervention Recovery Delta
    interventions = df[df.action != "ACCEPT"]

    deltas = []
    for _, row in interventions.iterrows():
        next_rows = df[
            (df.problem_id == row.problem_id) &
            (df.iteration > row.iteration)
        ].head(2)
        if len(next_rows) == 2:
            delta = next_rows.iloc[-1].accuracy - row.accuracy
            deltas.append(delta)

    print("Mean IRD:", np.mean(deltas) if deltas else 0)
