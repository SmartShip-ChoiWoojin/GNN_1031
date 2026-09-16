@echo off
cd /d "%~dp0"

set PY=python\python.exe
if not exist "%PY%" set PY=python

echo ============================================
echo FE Mesh Viewer - Diagnostic
echo ============================================
echo.
echo Collecting information. Please wait a moment...
echo.

"%PY%" src\diagnose.py > diagnostic_result.txt 2>&1

type diagnostic_result.txt

echo.
echo ============================================
echo Saved to file: diagnostic_result.txt
echo Please attach this file in your reply email.
echo ============================================
echo.
pause
