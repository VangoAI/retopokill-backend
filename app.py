import json
import uuid
import datetime
import sqlite3
import os
from dotenv import load_dotenv
import redshift_connector
from flask import Flask, request
from flask_cors import CORS

from pattern import Pattern
from patch import Patch

app = Flask(__name__)
CORS(app)

load_dotenv()
analytics_db_user = os.getenv('ANALYTICS_DB_USER')
analytics_db_password = os.getenv('ANALYTICS_DB_PASSWORD')

patches_cur = sqlite3.connect("pattern.db", check_same_thread=False).cursor()

@app.route('/get_expanded_patterns', methods=['POST'])
def get_expanded_patterns():
    sides = request.json
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


# analytics

@app.route('/get_uuid', methods=['GET'])
def get_uuid():
    return str(uuid.uuid4())

@app.route('/log_event', methods=['POST'])
def log_event():
    try:
        analytics_conn = redshift_connector.connect(
            host='default.815474491952.us-west-2.redshift-serverless.amazonaws.com',
            database='analytics',
            port=5439,
            user=analytics_db_user,
            password=analytics_db_password,
        )
        analytics_cur = analytics_conn.cursor()

        id = request.json['uuid']
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        event = request.json['event']

        analytics_cur.execute(f"INSERT INTO analytics VALUES (%s, %s, %s)", (id, timestamp, event))
        analytics_cur.close()

        analytics_conn.commit()
        analytics_conn.close()

        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == '__main__':
    app.run()
    