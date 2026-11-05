@echo off
chcp 65001 >nul
setlocal
rem 정책 판정관 설치 점검: 이 PC에서 제대로 돌아가는지 확인하고 outputs\doctor 에 보고서를 남김
rem   점검.bat          표준 점검(1~2분)
rem   점검.bat --full   묶음 파일 전체 대조까지(몇 분)
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%CD%\src"
set "PY="
if exist "runtime\python.exe" set "PY=runtime\python.exe"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY goto :no_python
"%PY%" -X utf8 -m jverdict.doctor %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%

:no_python
echo [안내] 실행 환경이 없습니다. 먼저 설치.bat 를 실행하세요.
echo.
pause
exit /b 1
