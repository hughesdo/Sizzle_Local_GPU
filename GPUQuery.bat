@echo off
REM ===========================================================================
REM  GPUQuery - what GPU is REALLY in this machine.
REM ===========================================================================
REM  Windows undersells this box badly: Win32_VideoController.AdapterRAM is a
REM  32-bit field, so any card with 4 GiB or more wraps to ~4.29 GB. Device
REM  Manager, dxdiag and Task Manager all read that field. On a pass-through
REM  card in a VM you also get a "Microsoft Hyper-V Video" adapter listed first.
REM
REM  This asks the driver and the CUDA runtime instead, and prints Windows'
REM  answer alongside so the difference is obvious rather than worrying.
REM ===========================================================================
setlocal
cd /d "%~dp0"

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo.
  echo   nvidia-smi was not found on your PATH.
  echo   No NVIDIA driver is installed, or it is not on PATH.
  echo.
  pause
  exit /b 1
)

REM The rich report needs the project venv (it asks PyTorch what it sees).
REM Without one, fall back to a plain nvidia-smi dump so this still tells you
REM something useful on a fresh clone.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\gpu_report.py
) else (
  echo.
  echo   [no .venv yet - showing the driver's raw view only]
  echo   Run install.bat for the full report, including what PyTorch sees.
  echo.
  nvidia-smi
)

echo.
pause
endlocal
