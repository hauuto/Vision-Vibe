@echo off
echo === Checking virtual environment ===

:: Nếu chưa có .venv thì tạo
if not exist .venv (
    echo Creating virtual environment...
    python3 -m venv .venv
)

:: Kích hoạt venv
call .venv\Scripts\activate.bat

echo === Installing requirements ===
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo === Starting Flask application ===
echo Opening browser in 3 seconds...

:: Start Flask app in background
start /B python3 app.py

:: Wait 3 seconds for Flask to start
timeout /t 3 /nobreak > nul

:: Open browser
start http://127.0.0.1:5000

echo Flask application is running at http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
pause
