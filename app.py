# write a flask app that has a post endpoint called /get_expanded_patterns
# that takes in a json object with the following fields:
#     - sides: a list of lists of 3-tuples of floats

import json
import sqlite3
from flask import Flask, request
from flask_cors import CORS
from pattern import Pattern
from patch import Patch

app = Flask(__name__)
CORS(app)

con = sqlite3.connect("pattern.db", check_same_thread=False)
cur = con.cursor()

@app.route('/get_expanded_patterns', methods=['POST'])
def get_expanded_patterns():
    sides = request.json
    patch = Patch(sides)

    res = cur.execute(f"SELECT topology, boundaryIDs FROM Patches WHERE nCorners='{len(sides)}' LIMIT 20")

    matching = []
    for row in res:
        topology_encoding, polychords_encoding = row[0], row[1]
        print("ENCODINGS:", topology_encoding, polychords_encoding)
        pattern = Pattern.from_encoding(topology_encoding, polychords_encoding)
        params = pattern.feasible(patch)
        print("PARAMS:", params)
        # if params is not None:
        #     expanded_pattern = pattern.expand(params)
        #     expanded_pattern.fit(patch)
        #     matching.append(expanded_pattern.to_json())
    return matching

if __name__ == '__main__':
    app.run()