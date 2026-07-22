@echo off
REM Launch the real-time driver-state demo (camera -> cascade -> classifier).
REM Usage:  run_live.bat  [--mirror] [--smooth 20] [--source 1] ...
setlocal
set "PYTHONPATH=%~dp0"
set "DMS_PY=C:\Users\Khongmeng\.conda\envs\dms-train\python.exe"
"%DMS_PY%" -m train.run_live %*
endlocal
