from flask import Flask, request, jsonify
import network, redis

app = Flask(__name__)

@app.route('/interaction/askProblem', methods=['POST'])
def process_request():
    questionJson = request.get_json()

    answer = network.get_solution(questionJson.get("question"))

    response = {'answer': answer[0],
                'problem': answer[1],
                'similarTopics': answer[2]}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)