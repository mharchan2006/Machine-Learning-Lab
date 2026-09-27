from flask import Blueprint, jsonify
from evaluate_test_set import evaluate_test_dataset

evaluate_bp = Blueprint('evaluate', __name__)

@evaluate_bp.route('/api/evaluate', methods=['GET', 'POST'])
def evaluate_route():
    try:
        metrics = evaluate_test_dataset()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
