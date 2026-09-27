from flask import Blueprint, jsonify
from populate_dataset_and_train import train_and_save_model

train_bp = Blueprint('train', __name__)

@train_bp.route('/api/train', methods=['POST'])
def train_route():
    try:
        res = train_and_save_model()
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
