@echo off
echo VibeMatch - Testing The Weeknd's "Open Hearts"
echo =========================================

if "%1"=="" (
    echo Error: Please provide the path to The Weeknd's song
    echo Usage: test_weeknd_song.bat path\to\open_hearts.mp3
    exit /b
)

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Testing song: %1
echo.

python test_weeknd.py %1

echo.
echo Testing complete!
echo To run the interactive demo: python demo.py %1

pause 