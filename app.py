from flask import Flask, request, render_template
from model_training import train_model
from chatbot import generate_response

app = Flask(__name__)


model, tokenizer = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get-response', methods=['POST'])
def get_response():
    input_text = request.form['user_input']
    response = generate_response(input_text, model, tokenizer)
    return response

if __name__ == '__main__':
    app.run(debug=True)
