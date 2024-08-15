from flask import Flask, request,jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer,util


app = Flask(__name__)
CORS(app)

model = SentenceTransformer("./conceptComparison")


@app.route("/probability",methods=['POST'])
def probability():
    data = request.get_json()
    cr = data.get('correct_reason')
    ur = data.get('user_reason')
    option = data.get('option_selected')
    questionId = data.get('questionId')
    correct_embedding = model.encode(cr, convert_to_tensor=True)
    student_embedding = model.encode(ur, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(correct_embedding, student_embedding)
    prob = cosine_similarity.item() * 100
    prob =  round(prob,2)
    print(prob)
    result = {
        "correct_reason":cr,
        "user_reason":ur,
        "prob":prob,
        "option_selected":option,
        "questionId":questionId
    }
    return jsonify(result),200


if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000)