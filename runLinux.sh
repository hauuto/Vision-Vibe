#!/bin/bash
echo "=== Checking virtual environment ==="

# Nếu chưa có .venv thì tạo
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Kích hoạt venv
source .venv/bin/activate

echo "=== Installing requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Starting Flask application ==="
echo "Opening browser in 3 seconds..."

# Start Flask app in background
python3 app.py &

# Wait 3 seconds for Flask to start
sleep 3

# Open browser (Linux khác nhau, hỗ trợ nhiều lệnh)
if command -v xdg-open &> /dev/null; then
    xdg-open http://127.0.0.1:5000
elif command -v gnome-open &> /dev/null; then
    gnome-open http://127.0.0.1:5000
elif command -v open &> /dev/null; then  # cho macOS
    open http://127.0.0.1:5000
else
    echo "Please open manually: http://127.0.0.1:5000"
fi

echo "Flask application is running at http://127.0.0.1:5000"
echo "Press Ctrl+C to stop the server"

# Giữ terminal chạy (để theo dõi log Flask)
wait
