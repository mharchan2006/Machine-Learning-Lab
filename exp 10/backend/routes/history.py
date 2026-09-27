from flask import Blueprint, jsonify
from backend.database import get_all_history, clear_all_history

history_bp = Blueprint('history', __name__)

@history_bp.route('/api/history', methods=['GET'])
def get_history():
    try:
        history = get_all_history()
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@history_bp.route('/api/history/clear', methods=['POST'])
def clear_history():
    try:
        clear_all_history()
        return jsonify({'success': True, 'message': 'History cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
