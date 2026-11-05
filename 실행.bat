@echo off
chcp 65001 >nul
setlocal
rem 정책 판정관 J-Verdict 화면 실행 (이 PC 안에서만 열림: http://localhost:8501)
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%CD%\src"
set "PY="
if exist "runtime\python.exe" set "PY=runtime\python.exe"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY goto :no_python
"%PY%" -X utf8 -c "import jverdict.ui, streamlit" >nul 2>&1
if errorlevel 1 goto :broken
echo 정책 판정관 화면을 엽니다. 처음 실행은 1~2분 걸릴 수 있습니다.
echo 브라우저가 열리지 않으면 아래에 나오는 주소를 직접 여세요. 끝내려면 이 창에서 Ctrl+C 를 누르세요.
echo.
"%PY%" -X utf8 -m jverdict.ui %*
if errorlevel 1 goto :fail
exit /b 0

:no_python
echo [안내] 실행 환경이 없습니다. 먼저 설치.bat 를 실행하세요.
goto :fail

:broken
echo [오류] 프로그램을 불러오지 못했습니다. 점검.bat 를 실행해 원인을 확인하세요.
goto :fail

:fail
echo.
pause
exit /b 1
