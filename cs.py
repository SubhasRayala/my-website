from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)

CSV_FILE = 'cs.csv'

# Ensure CSV has header and blank first row, with serial number starting
def init_csv():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            # First row blank
            writer.writerow([])
            # Header row: blank first column, then headers
            writer.writerow(['', 'Serial No', 'Name', 'Email', 'Subject', 'Query', 'DateTime'])

# Determine next serial number from existing file
def get_next_serial():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return 1
    with open(CSV_FILE, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
        # Skip first blank row and header row
        data_rows = rows[2:]
        if not data_rows:
            return 1
        # Serial numbers are in column 1 (index 1)
        last_serial_str = data_rows[-1][1]
        try:
            return int(last_serial_str) + 1
        except ValueError:
            return 1

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response

@app.route('/cs', methods=['POST', 'OPTIONS'])
def cs():
    if request.method == 'OPTIONS':
        return ('', 200)

    init_csv()  # make sure file + headers exist

    data = request.json

    serial_no = get_next_serial()

    # Format date like "JAN 12 10:20 AM"
    formatted_date = datetime.now().strftime('%b %d %I:%M%p, %Y').upper()


    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        # First column blank per your request, then serial and fields
        writer.writerow([
            '',
            serial_no,
            data.get("name", ""),
            data.get("email", ""),
            data.get("subject", ""),
            data.get("query", ""),
            formatted_date
        ])

    return jsonify(success=True), 200

if __name__ == '__main__':
    init_csv()
    app.run(debug=True)
