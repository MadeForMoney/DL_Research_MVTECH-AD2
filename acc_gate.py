import pickle
import numpy as np

# Load the trained model
with open('gbdt_gate.pkl', 'rb') as file:
    clf = pickle.load(file)
