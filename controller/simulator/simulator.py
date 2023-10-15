from flask import Flask, request
from generateData import generate_data

app = Flask(__name__)

@app.route('/generate-data', methods=['POST'])
def generate_data_endpoint():
    data = request.json 
    interval_count = data.get('interval_count')
    failure_type_ids = data.get('failure_type_ids')
    generate_data(interval_count, failure_type_ids)

if __name__ == '__main__':
    app.run(debug=True)