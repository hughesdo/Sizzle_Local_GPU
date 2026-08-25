@echo off
REM ===========================================================================
REM  GPUWatch - live GPU monitor. Leave this open while Sizzle renders.
REM ===========================================================================
REM  WHY THIS EXISTS: Windows Task Manager's GPU graph defaults to the "3D"
REM  engine, which stays near 0%% during CUDA compute. Sizzle's work shows up
REM  under the "Compute_0" engine instead - so Task Manager makes a fully
REM  pinned GPU look idle. In Task Manager you must click a graph's dropdown
REM  and choose Compute_0. Or just watch this window, which reads the driver
REM  directly and cannot be misleading.
REM
REM  Columns:
REM    util      %% of time the GPU had work running (want ~100 while rendering)
REM    mem_used  VRAM in use - the 22B model alone is ~41 GB in bf16
REM    pwr       watts drawn (the card's cap is printed at the top)
REM    sm_clk    core clock; drops when idle or thermally limited
REM    temp      degrees C
REM
REM  Ctrl-C to stop.  Refreshes every 2 seconds.
REM ===========================================================================
setlocal
cd /d "%~dp0"

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo   nvidia-smi not found on PATH - no NVIDIA driver installed?
  pause
  exit /b 1
)

echo.
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
echo.
echo   Task Manager's default "3D" graph will NOT show this work.
echo   Switch a graph to "Compute_0" there, or just watch below.
echo.
echo   time      util   mem_used / total      pwr      sm_clk   temp
echo   --------------------------------------------------------------------

:loop
for /f "tokens=1-6 delims=," %%a in ('nvidia-smi --query-gpu^=utilization.gpu^,memory.used^,memory.total^,power.draw^,clocks.sm^,temperature.gpu --format^=csv^,noheader^,nounits') do (
  echo   %time:~0,8%  %%a%%   %%b MiB / %%c MiB   %%d W   %%e MHz   %%f C
)
timeout /t 2 /nobreak >nul
goto loop

endlocal
