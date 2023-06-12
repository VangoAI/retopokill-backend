import sqlite3
from flask import Flask, request
from flask_cors import CORS

from pattern import Pattern
from patch import Patch

app = Flask(__name__)
CORS(app)

patches_cur = sqlite3.connect("pattern.db", check_same_thread=False).cursor()

@app.route('/get_expanded_patterns', methods=['POST'])
def get_expanded_patterns():
    sides = request.json['args']
    patch = Patch(sides)

    res = patches_cur.execute(f"SELECT topology, boundaryIDs FROM Patches WHERE nCorners='{len(sides)}' LIMIT 20")

    matching = []
    for row in res:
        topology_encoding, polychords_encoding = row[0], row[1]
        try:
            pattern = Pattern.from_encoding(topology_encoding, polychords_encoding)
            params = pattern.feasible(patch)
            if params is not None:
                expanded_pattern = pattern.expand(params)
                expanded_pattern.fit(patch)
                matching.append(expanded_pattern.to_json())
        except Exception as e:
            pass
    return matching

if __name__ == '__main__':
    app.run()
    