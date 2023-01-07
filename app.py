# write a flask app that has a post endpoint called /get_expanded_patterns
# that takes in a json object with the following fields:
#     - sides: a list of lists of 3-tuples of floats

import json
from flask import Flask, request
from flask_cors import CORS
from pattern import Pattern
from patch import Patch

app = Flask(__name__)
CORS(app)

# [3x3 grid, 2x5 star]
FAKE_DEMO_PATTERN_ENCODINGS = [
    '0102####03040405######0606070207######080408##05####07',
    '0102######0304########0403####',
]

PATTERNS = [Pattern(encoding) for encoding in FAKE_DEMO_PATTERN_ENCODINGS]

@app.route('/get_expanded_patterns', methods=['POST'])
def get_expanded_patterns():
    sides = request.json
    patch = Patch(sides)
    for pattern in PATTERNS:
        params = pattern.feasible(patch)
        if params is not None:
            expanded_pattern = pattern.expand(params)
            expanded_pattern.fit(patch)
            return expanded_pattern.to_json()
    return {}

if __name__ == '__main__':
    app.run()