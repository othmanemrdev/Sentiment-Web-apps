from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)



# === Load models and vectorizers ===
logreg_model_imdb = joblib.load("LogisticRegression_imdb.pkl")
logreg_vectorizer_imdb = joblib.load("vectorizer_logreg_imdb.pkl")
bayes_model_imdb = joblib.load("naive_bayes_imdb.pkl")
bayes_vectorizer_imdb = joblib.load("vectorizer_bayes_imdb.pkl")
tokenizer_imdb = BertTokenizer.from_pretrained('./bert_imdb/')
bert_model_imdb = BertForSequenceClassification.from_pretrained('./bert_imdb/')

logreg_model_yelp = joblib.load("LogisticRegression_yelp.pkl")
logreg_vectorizer_yelp = joblib.load("vectorizer_logreg_yelp.pkl")
bayes_model_yelp = joblib.load("naive_bayes_yelp.pkl")
bayes_vectorizer_yelp = joblib.load("vectorizer_bayes_yelp.pkl")
tokenizer_yelp = BertTokenizer.from_pretrained('./bert_yelp/')
bert_model_yelp = BertForSequenceClassification.from_pretrained('./bert_yelp/')

logreg_model_amazon = joblib.load("LogisticRegression_amazon.pkl")
logreg_vectorizer_amazon = joblib.load("vectorizer_logreg_amazon.pkl")
bayes_model_amazon = joblib.load("naive_bayes_amazon.pkl")
bayes_vectorizer_amazon = joblib.load("vectorizer_bayes_amazon.pkl")
tokenizer_amazon = BertTokenizer.from_pretrained('./bert_amazon/')
bert_model_amazon = BertForSequenceClassification.from_pretrained('./bert_amazon/')

logreg_model_merged = joblib.load("LogisticRegression_datasetmergd.pkl")
logreg_vectorizer_merged = joblib.load("vectorizer_logreg_merged.pkl")
bayes_model_merged = joblib.load("naive_bayes_datasetmerged.pkl")
bayes_vectorizer_merged = joblib.load("vectorizer_bayes_merged.pkl")
tokenizer_merged = BertTokenizer.from_pretrained('./bert_merged/')
bert_model_merged = BertForSequenceClassification.from_pretrained('./bert_merged/')

# === Prediction Helpers ===
def predict_with_bert(text, tokenizer, model, dataset="default"):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()

    if dataset == "yelp":
        if prediction == 0:
            return "Negative"
        elif prediction == 1:
            return "Positive"
        elif prediction == 2:
            return "Neutral"
    else:
        return "Positive" if prediction == 1 else "Negative"


# === IMDB Routes ===
@app.route('/predict-logreg-imdb', methods=['POST'])
def predict_logreg_imdb():
    text = request.json['text']
    X = logreg_vectorizer_imdb.transform([text])
    result = logreg_model_imdb.predict(X)[0]
    return jsonify({'result': "Positive" if result == 1 else "Negative"})

@app.route('/predict-bayes-imdb', methods=['POST'])
def predict_bayes_imdb():
    text = request.json['text']
    X = bayes_vectorizer_imdb.transform([text])
    result = bayes_model_imdb.predict(X)[0]
    return jsonify({'result': "Positive" if result == 1 else "Negative"})

@app.route('/predict-bert-imdb', methods=['POST'])
def predict_bert_imdb():
    text = request.json['text']
    result = predict_with_bert(text, tokenizer_imdb, bert_model_imdb)
    return jsonify({'result': result})

@app.route('/predict-all-imdb', methods=['POST'])
def predict_all_imdb():
    text = request.json['text']
    result_log = logreg_model_imdb.predict(logreg_vectorizer_imdb.transform([text]))[0]
    result_bay = bayes_model_imdb.predict(bayes_vectorizer_imdb.transform([text]))[0]
    result_bert = predict_with_bert(text, tokenizer_imdb, bert_model_imdb)
    return jsonify({
        'logreg': "Positive" if result_log == 1 else "Negative",
        'bayes': "Positive" if result_bay == 1 else "Negative",
        'bert': result_bert
    })

# === Yelp Routes ===
def decode_sentiment(value):
    if value == 0:
        return "Negative"
    elif value == 1:
        return "Positive"
    elif value == 2:
        return "Neutral"
    return "Unknown"

@app.route('/predict-logreg-yelp', methods=['POST'])
def predict_logreg_yelp():
    text = request.json['text']
    X = logreg_vectorizer_yelp.transform([text])
    result = logreg_model_yelp.predict(X)[0]
    return jsonify({'result': decode_sentiment(result)})

@app.route('/predict-bayes-yelp', methods=['POST'])
def predict_bayes_yelp():
    text = request.json['text']
    X = bayes_vectorizer_yelp.transform([text])
    result = bayes_model_yelp.predict(X)[0]
    return jsonify({'result': decode_sentiment(result)})


@app.route('/predict-bert-yelp', methods=['POST'])
def predict_bert_yelp():
    text = request.json['text']
    result = predict_with_bert(text, tokenizer_yelp, bert_model_yelp, dataset="yelp")
    return jsonify({'result': result})


@app.route('/predict-all-yelp', methods=['POST'])
def predict_all_yelp():
    text = request.json['text']
    result_log = logreg_model_yelp.predict(logreg_vectorizer_yelp.transform([text]))[0]
    result_bay = bayes_model_yelp.predict(bayes_vectorizer_yelp.transform([text]))[0]
    result_bert = predict_with_bert(text, tokenizer_yelp, bert_model_yelp, dataset="yelp")
    return jsonify({
        'logreg': decode_sentiment(result_log),
        'bayes': decode_sentiment(result_bay),
        'bert': result_bert
    })


# === Amazon Routes ===
@app.route('/predict-logreg-amazon', methods=['POST'])
def predict_logreg_amazon():
    text = request.json['text']
    X = logreg_vectorizer_amazon.transform([text])
    result = logreg_model_amazon.predict(X)[0]
    return jsonify({'result': "Positive" if result == 1 else "Negative"})

@app.route('/predict-bayes-amazon', methods=['POST'])
def predict_bayes_amazon():
    text = request.json['text']
    X = bayes_vectorizer_amazon.transform([text])
    result = bayes_model_amazon.predict(X)[0]
    return jsonify({'result': "Positive" if result == 1 else "Negative"})

@app.route('/predict-bert-amazon', methods=['POST'])
def predict_bert_amazon():
    text = request.json['text']
    result = predict_with_bert(text, tokenizer_amazon, bert_model_amazon)
    return jsonify({'result': result})

@app.route('/predict-all-amazon', methods=['POST'])
def predict_all_amazon():
    text = request.json['text']
    result_log = logreg_model_amazon.predict(logreg_vectorizer_amazon.transform([text]))[0]
    result_bay = bayes_model_amazon.predict(bayes_vectorizer_amazon.transform([text]))[0]
    result_bert = predict_with_bert(text, tokenizer_amazon, bert_model_amazon)
    return jsonify({
        'logreg': "Positive" if result_log == 1 else "Negative",
        'bayes': "Positive" if result_bay == 1 else "Negative",
        'bert': result_bert
    })
    
# === IMDB Merged ===
@app.route('/predict-logreg-merged', methods=['POST'])
def predict_logreg_merged():
    text = request.json['text']
    X = logreg_vectorizer_merged.transform([text])
    result = logreg_model_merged.predict(X)[0]
    return jsonify({'result': "Positive" if result == 1 else "Negative"})

@app.route('/predict-bayes-merged', methods=['POST'])
def predict_bayes_merged():
    text = request.json['text']
    X = bayes_vectorizer_merged.transform([text])
    result = bayes_model_merged.predict(X)[0]
    return jsonify({'result': "Positive" if result == 1 else "Negative"})

@app.route('/predict-bert-merged', methods=['POST'])
def predict_bert_merged():
    text = request.json['text']
    result = predict_with_bert(text, tokenizer_merged, bert_model_merged)
    return jsonify({'result': result})

@app.route('/predict-all-merged', methods=['POST'])
def predict_all_merged():
    text = request.json['text']
    result_log = logreg_model_merged.predict(logreg_vectorizer_merged.transform([text]))[0]
    result_bay = bayes_model_merged.predict(bayes_vectorizer_merged.transform([text]))[0]
    result_bert = predict_with_bert(text, tokenizer_merged, bert_model_merged)
    return jsonify({
        'logreg': "Positive" if result_log == 1 else "Negative",
        'bayes': "Positive" if result_bay == 1 else "Negative",
        'bert': result_bert
    })


# === Merged Prediction Route ===
@app.route('/predict-all', methods=['POST'])
def predict_all():
    text = request.json['text']
    return jsonify({
        'imdb': predict_all_imdb().get_json(),
        'yelp': predict_all_yelp().get_json(),
        'amazon': predict_all_amazon().get_json(),
        'merged': predict_all_merged().get_json()
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)