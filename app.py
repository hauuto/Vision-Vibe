from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hello')
def hello_world():
    return jsonify({'message': 'Hello World from Flask!'})

if __name__ == '__main__':
    app.run(debug=True)
