from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hello')
def hello_world():
    return jsonify({'message': 'Hello World from Flask!'})

@app.route('/bt1')
def bt1():
    return 'hello bt1 nha'

# Template context processor để có thể sử dụng include trong template
@app.context_processor
def inject_template_vars():
    return dict()

if __name__ == '__main__':
    app.run(debug=True)
