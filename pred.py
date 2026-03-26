"""
pred.py - Painting classifier for CSC311 ML Challenge
Predicts which of three paintings a student described:
  - The Persistence of Memory (Dali)
  - The Starry Night (Van Gogh)
  - The Water Lily Pond (Monet)

Model: Logistic Regression trained with sklearn, weights embedded as numpy arrays.
Features: Likert emotion scales, numeric responses, season/food/room/description keywords.
CV accuracy: ~87%
"""

import sys
import csv
import numpy


# ── Embedded model weights (trained on full dataset, C=5.0 LR) ──────────────

CLASSES = ['The Persistence of Memory', 'The Starry Night', 'The Water Lily Pond']

WEIGHTS = numpy.array([
    [0.5305219181750178, -0.7578899347972484, -0.07559862443166689,
     0.5138131751915639, -0.02686477010230074, 0.11168589907085641,
     0.1653044315331133, -0.7912533936716151, 0.07301157148072349,
     1.575007276174887, 0.19562569278404796, -0.8313607012777933,
     -1.639669192073284, -0.41127885193770874, 0.5097688173100884,
     0.2968476875761349, -1.4499931455233432, -0.31071927265988286,
     -0.629017463656753, 1.6026516235381785, -0.03211922296408986,
     -0.48666398116486653],
    [-0.06034831978029328, 0.04068748430416892, 0.08072096008809414,
     -0.045769297395686984, 0.030349554182901927, -0.10937585442025796,
     0.05901000277194644, -0.5501527154426746, 0.12080508029206101,
     0.11918346049942116, 2.276030066671328, -0.1016625272963696,
     -0.17039647142233713, 0.6558911688619536, 0.14804906795104858,
     -0.15198254635744618, 1.0671616584675354, -0.24369711538210376,
     0.3461206825653134, -0.3320419853687113, -0.07800002411376186,
     1.3727803368720466],
    [-0.4701735983949195, 0.7172024504930067, -0.005122335656273403,
     -0.46804387779594714, -0.0034847840809337297, -0.00231004465064638,
     -0.22431443430552314, 1.3414061091142802, -0.19381665177276272,
     -1.6941907366743425, -2.4716557594553423, 0.9330232285741378,
     1.810065663495646, -0.24461231692424742, -0.6578178852611319,
     -0.14486514121868474, 0.38283148705583386, 0.554416388041993,
     0.282896781091427, -1.2706096381694272, 0.11011924707785534,
     -0.8861163557071883]
])

INTERCEPT = numpy.array([-0.6710807417608232, 0.1434892035844533, 0.5275915381763615])


# ── Feature extraction ───────────────────────────────────────────────────────

LIKERT_MAP = {
    '1 - Strongly disagree': 1.0,
    '2 - Disagree': 2.0,
    '3 - Neutral/Unsure': 3.0,
    '4 - Agree': 4.0,
    '5 - Strongly agree': 5.0,
}

EMOTION_COLS = [
    'This art piece makes me feel sombre.',
    'This art piece makes me feel content.',
    'This art piece makes me feel calm.',
    'This art piece makes me feel uneasy.',
]

FOOD_WL_KEYWORDS  = ['salad', 'watermelon', 'green', 'strawberry', 'cucumber',
                      'apple', 'fruit', 'vegetable', 'melon']
FOOD_SN_KEYWORDS  = ['ice cream', 'blueberry', 'pasta', 'spaghetti', 'steak',
                      'cake', 'sushi', 'blue']
FOOD_POM_KEYWORDS = ['pizza', 'bread', 'cheese', 'soup', 'banana', 'noodle',
                     'stale', 'lasagna', 'fries']

ROOM_KEYS = ['office', 'bedroom', 'bathroom', 'living']

DESC_MELANCHOLY = ['time', 'melt', 'clock', 'surreal', 'dread', 'anxious',
                   'dream', 'eerie', 'strange']
DESC_PEACEFUL   = ['peace', 'calm', 'serene', 'nature', 'relax', 'tranquil',
                   'garden', 'water', 'pond', 'lily']
DESC_AWE        = ['wonder', 'awe', 'inspire', 'sky', 'star', 'night',
                   'vast', 'infinite', 'beautiful']


def _safe_float(val, default):
    """Parse a value to float, returning default on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _has_keyword(text, keywords):
    """Return 1.0 if any keyword is found in text, else 0.0."""
    return 1.0 if any(k in text for k in keywords) else 0.0


def extract_features(row):
    """
    Extract a 22-dimensional feature vector from a single CSV row (dict).
    """
    features = []

    # 1. Likert emotion scales (4 features)
    for col in EMOTION_COLS:
        val = row.get(col, '')
        features.append(LIKERT_MAP.get(val, 3.0))

    # 2. Numeric survey responses (3 features)
    intensity = _safe_float(row.get('On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?', ''), 6.0)
    features.append(intensity)

    colours = _safe_float(row.get('How many prominent colours do you notice in this painting?', ''), 3.0)
    features.append(min(colours, 20.0))

    objects = _safe_float(row.get('How many objects caught your eye in the painting?', ''), 3.0)
    features.append(min(objects, 20.0))

    # 3. Season keywords (5 features: spring, summer, fall, winter, missing)
    season = row.get('What season does this art piece remind you of?', '') or ''
    for s in ['Spring', 'Summer', 'Fall', 'Winter']:
        features.append(1.0 if s in season else 0.0)
    features.append(1.0 if season == '' else 0.0)

    # 4. Food keywords (3 features)
    food = (row.get('If this painting was a food, what would be?', '') or '').lower()
    features.append(_has_keyword(food, FOOD_WL_KEYWORDS))
    features.append(_has_keyword(food, FOOD_SN_KEYWORDS))
    features.append(_has_keyword(food, FOOD_POM_KEYWORDS))

    # 5. Room keywords (4 features)
    room = (row.get('If you could purchase this painting, which room would you put that painting in?', '') or '').lower()
    for r in ROOM_KEYS:
        features.append(1.0 if r in room else 0.0)

    # 6. Description keywords (3 features)
    desc = (row.get('Describe how this painting makes you feel.', '') or '').lower()
    features.append(_has_keyword(desc, DESC_MELANCHOLY))
    features.append(_has_keyword(desc, DESC_PEACEFUL))
    features.append(_has_keyword(desc, DESC_AWE))

    return numpy.array(features)


# ── Inference ────────────────────────────────────────────────────────────────

def softmax(z):
    """Numerically stable softmax."""
    e = numpy.exp(z - numpy.max(z))
    return e / e.sum()


def predict(row):
    """
    Predict painting label for a single row (dict from csv.DictReader).
    Returns one of: 'The Persistence of Memory', 'The Starry Night', 'The Water Lily Pond'
    """
    x = extract_features(row)
    logits = WEIGHTS @ x + INTERCEPT
    probs = softmax(logits)
    return CLASSES[int(numpy.argmax(probs))]


def predict_all(filename):
    """
    Read CSV file and return a list of predictions (one per row).
    """
    data = csv.DictReader(open(filename, encoding='utf-8'))
    predictions = []
    for row in data:
        predictions.append(predict(row))
    return predictions


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pred.py <test_csv_file>")
        sys.exit(1)
    preds = predict_all(sys.argv[1])
    for p in preds:
        print(p)