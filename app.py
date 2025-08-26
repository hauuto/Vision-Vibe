from flask import Flask, render_template, jsonify
from flask_frozen import Freezer
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/slide1/')
def slide01():
    return render_template('components/slide01.html')

# --- API DEMOS ---

@app.route('/api/hello/')
def api_hello():
    return jsonify({'message': 'Hello from Flask API!'})


# Template context processor để có thể sử dụng include trong template
@app.context_processor
def inject_template_vars():
    return dict()

freezer = Freezer(app)
app.config['FREEZER_DESTINATION'] = 'docs'
if __name__ == '__main__':
    freezer.freeze()
    app.run(debug=True)
