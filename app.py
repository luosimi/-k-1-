import os
import threading
import logging
import subprocess
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory
from config import config, save_config, CONFIG_FILE
from core.indexer import Indexer
from core.search_engine import SearchEngine
from core.feature_extractor import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global instances
search_engine = SearchEngine()
feature_extractor = FeatureExtractor() # This will load the model (lazy load suggested but here we load on start or first use)

# Sync status
sync_status = {
    "is_syncing": False,
    "progress": 0,
    "total": 0,
    "message": "Idle"
}
sync_lock = threading.Lock()

def init_app():
    """Initialize search engine on startup."""
    index_path = config.get("index_path", "./data/index.csv")
    if os.path.exists(index_path):
        search_engine.load_index(index_path)

@app.route('/')
def index():
    return render_template('index.html', config=config)

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        data = request.json
        if 'gallery_path' in data:
            if os.path.isdir(data['gallery_path']):
                config['gallery_path'] = data['gallery_path']
                save_config(config)
                return jsonify({"status": "success", "message": "Configuration saved."})
            else:
                return jsonify({"status": "error", "message": "Invalid directory path."}), 400
    return jsonify(config)

@app.route('/api/select-folder', methods=['POST'])
def select_folder():
    try:
        # Run a subprocess to open the dialog, isolated from Flask's threads
        cmd = [
            sys.executable, 
            '-c', 
            "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True); print(filedialog.askdirectory())"
        ]
        
        # Windows-specific: Hide console window to avoid popping up empty CMD
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=startupinfo)
        
        path = result.stdout.strip()
        
        if path:
            return jsonify({"status": "success", "path": path})
        else:
            return jsonify({"status": "cancel", "message": "No folder selected."})
            
    except Exception as e:
        logger.error(f"Select folder error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def run_sync_task(gallery_path, index_path):
    global sync_status
    try:
        indexer = Indexer(gallery_path, index_path)
        
        def progress_callback(current, total):
            sync_status['progress'] = current
            sync_status['total'] = total
            sync_status['message'] = f"Processing {current}/{total}"

        indexer.sync(progress_callback=progress_callback)
        
        # Reload search engine after sync
        search_engine.load_index(index_path)
        
        with sync_lock:
            sync_status['is_syncing'] = False
            sync_status['message'] = "Sync completed."
            sync_status['progress'] = 0
            sync_status['total'] = 0
            
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        with sync_lock:
            sync_status['is_syncing'] = False
            sync_status['message'] = f"Sync failed: {str(e)}"

@app.route('/api/sync', methods=['POST'])
def start_sync():
    global sync_status
    with sync_lock:
        if sync_status['is_syncing']:
            return jsonify({"status": "error", "message": "Sync already in progress."}), 409
        
        sync_status['is_syncing'] = True
        sync_status['message'] = "Starting sync..."
        sync_status['progress'] = 0
        sync_status['total'] = 0

    gallery_path = config.get('gallery_path')
    index_path = config.get('index_path')
    
    if not gallery_path or not os.path.exists(gallery_path):
        with sync_lock:
            sync_status['is_syncing'] = False
            sync_status['message'] = "Gallery path invalid."
        return jsonify({"status": "error", "message": "Invalid gallery path."}), 400

    thread = threading.Thread(target=run_sync_task, args=(gallery_path, index_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "success", "message": "Sync started."})

@app.route('/api/status')
def get_status():
    status = sync_status.copy()
    # Add device info
    try:
        status['device_info'] = feature_extractor.model.get_device_info()
    except Exception:
        status['device_info'] = {"device": "unknown", "name": "Initializing..."}
    return jsonify(status)

@app.route('/api/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400

    try:
        # Read image from stream
        from PIL import Image
        img = Image.open(file.stream)
        
        # Extract feature
        query_vec = feature_extractor.extract_from_image(img)
        
        # Search
        results = search_engine.search(query_vec, top_k=50)
        
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    gallery_path = config.get('gallery_path')
    # Security check: ensure the file is within gallery_path is handled by send_from_directory roughly,
    # but we should be careful about '..' in filename. send_from_directory handles basic traversal attacks.
    return send_from_directory(gallery_path, filename)

if __name__ == '__main__':
    # Initialize index on start
    init_app()
    app.run(debug=True, port=5000)
