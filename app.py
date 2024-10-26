from flask import Flask, request, jsonify
from flask_cors import CORS  # Импортируем CORS
import network

app = Flask(__name__)

# Настраиваем CORS для разрешения запросов с localhost:3000
CORS(app, resources={r"/interaction/*": {"origins": "http://localhost:3000"}})

@app.route('/interaction/askProblem', methods=['POST'])
def process_request():
    questionJson = request.get_json()

    answer = network.get_solution(questionJson.get("question"))

    response = {
        'solution': answer[0],
        'label': answer[1],
        'percentages': answer[2],
        'similarTopics': answer[3]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
