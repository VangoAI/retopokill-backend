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

# ALL NUMBERS IN HEX
FAKE_DEMO_PATTERN_ENCODINGS = [
    '00000000000000010000000000000002################################0000000000000003000000000000000400000000000000040000000000000005################################################00000000000000060000000000000006000000000000000700000000000000020000000000000007################################################000000000000000800000000000000040000000000000008################0000000000000005################################0000000000000007', # 3x3 grid
    # '0102######0304########0403####', # 2x2x2x2x2 star
    # '0102####03040405####06040307020708######090A0B050B########0A09##0B0A##08', # 4x4x2x4
    # '0102####03040506######0708090504090A0B######0C08070A09080A0509080B0A0C######0B', # 3x3 more complex grid
    # '01####020304##05060607080809######0605070306##0807090408####', # 3x3x3x1
    # '0102####030404########0505##02####04', # 3x2x3x2 grid
    # '0102######0304######05040305######04', # 2x2x3x3
    # '01####020304####030205040305######04', # 1x2x2x3
]

#PATTERNS = [Pattern(encoding) for encoding in FAKE_DEMO_PATTERN_ENCODINGS]

@app.route('/get_expanded_patterns', methods=['POST'])
def get_expanded_patterns():
    sides = request.json
    patch = Patch(sides)
    matching = []

    res = cur.execute(f"SELECT topology FROM Patches WHERE nCorners='{len(sides)}' LIMIT 100")
    i = 0
    for row in res:
        print(i, row)
        i += 1
        
        encoding = row[0]
        try:
            pattern = Pattern(encoding)
            params = pattern.feasible(patch)
            if params is not None:
                expanded_pattern = pattern.expand(params)
                expanded_pattern.fit(patch)
                matching.append(expanded_pattern.to_json())
        except Exception as e:
            print("error:", i)
    print(matching)
    return matching

if __name__ == '__main__':
    app.run()