import pickle
import numpy as np

# Load the XGBoost model
with open("model/optimized_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Map model output indices to gesture labels
index_to_label = {
    0: 'call', 1: 'dislike', 2: 'fist', 3: 'four', 4: 'like',
    5: 'mute', 6: 'ok', 7: 'one', 8: 'palm', 9: 'peace',
    10: 'peace_inverted', 11: 'rock', 12: 'stop', 13: 'stop_inverted',
    14: 'three', 15: 'three2', 16: 'two_up', 17: 'two_up_inverted',
}

# Map chosen gestures to directions
gesture_to_direction = {
    'like': 'up',
    'dislike': 'down',
    'ok': 'right',
    'peace': 'left',
}

def predict_direction(features: list) -> str:
    features = np.array(features).reshape(1, -1)
    pred_class = model.predict(features)[0]
    label = index_to_label.get(pred_class, "unknown")
    return gesture_to_direction.get(label, "unknown")
