import os
import sys

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, send_from_directory, jsonify
from backend.config import FRONTEND_DIR, UPLOADS_DIR, RESULTS_DIR, BASE_DIR
from backend.database import init_db
from backend.routes.predict import predict_bp
from backend.routes.train import train_bp
from backend.routes.evaluate import evaluate_bp
from backend.routes.history import history_bp

app = Flask(__name__, static_folder=FRONTEND_DIR)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Register blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(train_bp)
app.register_blueprint(evaluate_bp)
app.register_blueprint(history_bp)

# Static file serving routes
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return jsonify({'error': 'Not found'}), 404

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOADS_DIR, filename)

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route('/api/health')
def health():
    return jsonify({'status': 'online', 'service': 'MelanomaClassifier'})

if __name__ == '__main__':
    init_db()
    print("================================================================")
    print(" Melanoma Skin Lesion Classification System (Flask Server) ")
    print(" Listening on http://127.0.0.1:5000")
    print(" Open your web browser to http://127.0.0.1:5000")
    print("================================================================")
    app.run(host='0.0.0.0', port=5000, debug=True)
