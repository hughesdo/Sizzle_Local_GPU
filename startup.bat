@echo off
REM ===========================================================================
REM  SIZZLE - run LOCALLY on this machine's GPU.  Double-click me.
REM ===========================================================================
REM  Opens http://127.0.0.1:8000 in your browser. Nothing is exposed to the
REM  internet and nothing is uploaded anywhere: the LTX-2.3 model runs right
REM  here on your NVIDIA card.
REM
REM  To share it with a friend over a Cloudflare tunnel, use host.bat instead.
REM  First time here? Run install.bat once.
REM ===========================================================================
setlocal
cd /d "%~dp0"

set "SIZZLE_HOST=127.0.0.1"
set "SIZZLE_PORT=8000"

REM Prefer the project venv that install.bat created.
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo !! No .venv found - run install.bat once first.
  echo.
  pause
  exit /b 1
)

REM --- are the weights actually here? -------------------------------------
REM  Checking one checkpoint by hand is not enough: a render needs the
REM  checkpoint AND both LoRAs AND the upscaler AND the Gemma text encoder, and
REM  missing any one of them fails at Generate rather than at startup. Ask the
REM  app itself. We do NOT block on it - the timeline, the waveform and
REM  auto-prompt all work without weights, and the UI badge says what is short.
"%PY%" scripts\check_weights.py
if errorlevel 1 (
  echo.
  echo    ------------------------------------------------------------------
  echo     GENERATE will not work until the missing weights above are on disk.
  echo     Everything else in the app works right now.
  echo    ------------------------------------------------------------------
  echo.
)

echo.
echo ^>^> starting SIZZLE on http://%SIZZLE_HOST%:%SIZZLE_PORT%
echo    rendering locally - no fal.ai, no Cloudinary, no uploads
echo    (Ctrl-C in this window to stop)
echo.

REM open the UI in the default browser a moment after the server boots
start "" cmd /c "timeout /t 3 >nul & start http://%SIZZLE_HOST%:%SIZZLE_PORT%/"

"%PY%" -m uvicorn backend.app:app --host %SIZZLE_HOST% --port %SIZZLE_PORT%

endlocal
