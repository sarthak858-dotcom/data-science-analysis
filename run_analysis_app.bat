@echo off
cd /d "%~dp0"
python -m streamlit run data_analysis_app.py --server.address 127.0.0.1
if errorlevel 1 (
    echo.
    echo Could not start the app. Install its packages with:
    echo python -m pip install -r requirements.txt
    pause
)
