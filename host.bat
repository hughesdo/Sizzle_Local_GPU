@echo off
REM ===========================================================================
REM  SIZZLE - SERVE IT TO FRIENDS over a Cloudflare quick tunnel.
REM ===========================================================================
REM  Starts the local server and opens a temporary, random public URL
REM  (https://something.trycloudflare.com). Hand that URL to your friends and
REM  they can build timelines and render in their browser.
REM  Close this window to stop serving - the URL dies with it.
REM
REM  WORTH KNOWING: every render runs on YOUR GPU, one at a time (the app holds
REM  a global single-render lock, so two friends cannot fight over the card).
REM  It costs you no API credits any more - just your electricity and your GPU
REM  being busy. The random URL is your only access control, so share it with
REM  people you trust and close the window when you are done.
REM
REM  Requires cloudflared on your PATH:
REM    https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
REM ===========================================================================
setlocal
cd /d "%~dp0"

set "SIZZLE_HOST=127.0.0.1"
set "SIZZLE_PORT=8000"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not exist ".venv\Scripts\python.exe" (
  echo !! No .venv found - run install.bat once first.
  pause
  exit /b 1
)

where cloudflared >nul 2>nul
if errorlevel 1 (
  echo.
  echo !! cloudflared was not found on your PATH.
  echo    Install it, then run host.bat again:
  echo    https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
  echo.
  echo    On Windows the quickest way is:
  echo        winget install --id Cloudflare.cloudflared
  echo.
  pause
  exit /b 1
)

REM Ask the app what it is actually missing (see scripts/check_weights.py).
REM Worth pausing on here: handing friends a URL whose GENERATE button fails is
REM a worse experience than making them wait for the download to finish.
"%PY%" scripts\check_weights.py
if errorlevel 1 (
  echo.
  echo    ------------------------------------------------------------------
  echo     Your friends will be able to build timelines, but GENERATE will FAIL
  echo     until the weights above finish downloading. Run:
  echo         .venv\Scripts\python.exe scripts\download_models.py
  echo    ------------------------------------------------------------------
  echo.
  echo     Press Ctrl-C to stop, or any key to share it anyway.
  pause
)

REM Warm the model before anyone connects, so the first friend to hit GENERATE
REM does not sit through a 46 GB checkpoint load wondering if it is broken.
set "SIZZLE_WARMUP=1"

echo ^>^> starting SIZZLE server (background) on http://%SIZZLE_HOST%:%SIZZLE_PORT%
start "sizzle-server" /min "%PY%" -m uvicorn backend.app:app --host %SIZZLE_HOST% --port %SIZZLE_PORT%

REM Give uvicorn a moment to bind the port before cloudflared probes it.
timeout /t 4 >nul

echo ^>^> opening Cloudflare quick tunnel...
echo.
echo    ============================================================
echo     Your public URL appears below as:
echo         https://SOMETHING.trycloudflare.com
echo     Share THAT link with your friends.
echo     Close this window when you're done - the link then dies.
echo    ============================================================
echo.

cloudflared tunnel --url http://%SIZZLE_HOST%:%SIZZLE_PORT%

REM cloudflared exited (window closed / Ctrl-C): stop the background server too.
echo.
echo ^>^> tunnel closed - shutting the Sizzle server down
taskkill /FI "WINDOWTITLE eq sizzle-server*" /F >nul 2>nul
endlocal
