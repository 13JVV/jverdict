@echo off
chcp 65001 >nul
setlocal
rem 정책 판정관 설치
rem  - 실행 환경(runtime 폴더)이 들어 있는 묶음: 설치할 것이 없으므로 점검만 한다.
rem  - 실행 환경이 없는 묶음: 이 PC의 Python 3.11 이상으로 .venv 를 만들고 패키지를 내려받는다(인터넷 필요).
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
if exist "runtime\python.exe" goto :bundled

set "SYS="
for %%C in ("py -3.14" "py -3" "python") do call :try_python %%C
if not defined SYS goto :no_python
echo 사용할 Python:
%SYS% --version
if exist ".venv\Scripts\python.exe" goto :venv_ready
echo 가상환경(.venv)을 만듭니다...
%SYS% -m venv .venv
if errorlevel 1 goto :venv_fail

:venv_ready
set "VPY=.venv\Scripts\python.exe"
echo 패키지를 내려받아 설치합니다(몇 분 걸립니다)...
"%VPY%" -m pip install --disable-pip-version-check -r requirements\lock-win-cp314.txt
if not errorlevel 1 goto :installed
echo [안내] 고정 버전 설치에 실패해 호환 버전으로 다시 설치합니다(판정 수치가 마지막 자리에서 다를 수 있음).
"%VPY%" -m pip install --disable-pip-version-check -r requirements\base.txt
if errorlevel 1 goto :pip_fail

:installed
echo.
echo 설치가 끝났습니다. 설치 점검을 실행합니다.
set "PYTHONPATH=%CD%\src"
"%VPY%" -X utf8 -m jverdict.doctor
goto :done

:bundled
echo 이 폴더에는 실행 환경(runtime)이 이미 들어 있어 설치할 것이 없습니다. 설치 점검을 실행합니다.
"runtime\python.exe" -X utf8 -m jverdict.doctor
goto :done

:done
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%

:try_python
if defined SYS exit /b 0
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "SYS=%~1"
exit /b 0

:no_python
echo [오류] Python 3.11 이상을 찾지 못했습니다.
echo        실행 환경이 들어 있는 묶음(JVerdict_win64_*.zip)을 쓰거나, python.org 에서 Python을 설치한 뒤 다시 실행하세요.
goto :fail

:venv_fail
echo [오류] 가상환경(.venv)을 만들지 못했습니다.
goto :fail

:pip_fail
echo [오류] 패키지 설치에 실패했습니다. 인터넷 연결을 확인하세요.
goto :fail

:fail
echo.
pause
exit /b 1
