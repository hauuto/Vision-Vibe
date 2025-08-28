#!/bin/bash

echo "Installing requirements..."
pip install -r requirements.txt

echo "Starting Flask application..."
echo "Opening browser in 3 seconds..."

# Start Flask app in background
python app.py &
FLASK_PID=$!

# Wait 3 seconds for Flask to start
sleep 3

# Open browser (works on Linux/Mac/WSL)
if command -v xdg-open > /dev/null; then
    xdg-open http://127.0.0.1:5000
elif command -v open > /dev/null; then
    open http://127.0.0.1:5000
else
    echo "Please open http://127.0.0.1:5000 in your browser"
fi

echo "Flask application is running at http://127.0.0.1:5000"
echo "Press Ctrl+C to stop the server"

# Wait for Flask process
wait $FLASK_PID
