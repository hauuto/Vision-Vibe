from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/essay_1')
def essay_1():
    return render_template('Essay_1.html')

# Template context processor để có thể sử dụng include trong template
@app.context_processor
def inject_template_vars():
    return dict()

if __name__ == '__main__':
    app.run(debug=True)
