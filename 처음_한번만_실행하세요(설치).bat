@echo off
chcp 65001 >nul
setlocal
title [Erickson AI] 자동 설치 시스템
echo ======================================================
echo [Erickson AI] 에릭슨 아카데미 자동 설치를 시작합니다.
echo ======================================================
echo.

:: 1. 파이썬 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [안내] 파이썬을 찾을 수 없습니다. 자동으로 설치를 시도합니다...
    winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
    echo [중요] 설치 후 인식을 위해 창이 닫힙니다. 다시 실행해 주세요!
    pause
    exit
)

:: 2. 라이브러리 설치
echo [안내] 필요한 프로그램들을 설치 중입니다...
python -m pip install --upgrade pip
python -m pip install -r "%~dp0requirements.txt"

echo.
echo [성공] 모든 설치가 완료되었습니다!
pause
