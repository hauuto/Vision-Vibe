@echo off
echo Installing requirements...
python -m pip install -r requirements.txt

echo Starting Flask application...
echo Opening browser in 3 seconds...

:: Start Flask app in background
start /B python app.py

:: Wait 3 seconds for Flask to start
timeout /t 3 /nobreak > nul

:: Open browser
start http://127.0.0.1:5000

echo Flask application is running at http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
pause