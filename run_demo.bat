@echo off
echo VibeMatch - Interactive Demo
echo =========================================

if "%1"=="" (
    echo Error: Please provide the path to The Weeknd's song
    echo Usage: run_demo.bat path\to\open_hearts.mp3
    exit /b
)

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting interactive demo...
echo.

python demo.py %1

echo.
echo Demo complete!

pause 