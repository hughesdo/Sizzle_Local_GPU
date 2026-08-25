"""
The real GPU on this machine - and why Windows disagrees.

Windows' own reporting is not trustworthy on a pass-through card like this one:
`Win32_VideoController.AdapterRAM` is a 32-bit field, so anything at or above
4 GiB wraps and gets reported as ~4.29 GB. On this box that turns 96 GB of
Blackwell into "4 GB" in Device Manager, dxdiag and Task Manager. The driver and
CUDA runtime know the truth; this script asks them instead, and shows Windows'
answer side by side so the discrepancy is obvious rather than alarming.

Run it through GPUQuery.bat, or directly:
    .venv\\Scripts\\python.exe scripts\\gpu_report.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys

BAR = "=" * 72


def _run(cmd: list[str]) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return p.stdout if p.returncode == 0 else None
    except Exception:
        return None


def _head(title: str) -> None:
    print()
    print(BAR)
    print(f"  {title}")
    print(BAR)


def nvidia_smi() -> dict:
    """Ground truth, straight from the driver."""
    fields = [
        "name", "driver_version", "vbios_version", "memory.total", "memory.used",
        "memory.free", "compute_cap", "pcie.link.gen.max", "pcie.link.width.max",
        "power.limit", "power.draw", "temperature.gpu", "utilization.gpu",
        "clocks.max.sm", "serial", "uuid",
    ]
    out = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits"])
    if not out or not out.strip():
        return {}
    values = [v.strip() for v in out.strip().splitlines()[0].split(",")]
    return dict(zip(fields, values))


def cuda_version() -> str | None:
    out = _run(["nvidia-smi", "--query"])
    if not out:
        return None
    for line in out.splitlines():
        if "CUDA Version" in line:
            return line.split(":", 1)[1].strip()
    return None


def virtualization() -> str | None:
    out = _run(["nvidia-smi", "--query"])
    if not out:
        return None
    for line in out.splitlines():
        if "Virtualization Mode" in line and ":" in line:
            val = line.split(":", 1)[1].strip()
            if val and val != "N/A":
                return val
    return None


def windows_view() -> list[dict]:
    """What Windows itself believes. Included precisely because it is wrong."""
    if not sys.platform.startswith("win"):
        return []
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        return []
    out = _run([ps, "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | "
                "ForEach-Object { $_.Name + '|' + $_.AdapterRAM + '|' + $_.DriverVersion }"])
    if not out:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            rows.append({"name": parts[0].strip(), "ram": parts[1].strip(),
                         "driver": parts[2].strip()})
    return rows


def torch_view() -> dict:
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {"torch": torch.__version__, "cuda": False}
    p = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "cuda": True,
        "name": p.name,
        "cc": f"sm_{p.major}{p.minor}",
        "sms": p.multi_processor_count,
        "vram_bytes": p.total_memory,
        "arch_list": torch.cuda.get_arch_list(),
        "bf16": torch.cuda.is_bf16_supported(),
    }


def main() -> int:
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found - no NVIDIA driver on PATH.")
        print("Sizzle needs an NVIDIA GPU to render locally (or SIZZLE_BACKEND=mock).")
        return 1

    smi = nvidia_smi()
    if not smi:
        print("nvidia-smi found but returned nothing usable.")
        return 1

    total_mib = float(smi.get("memory.total") or 0)
    used_mib = float(smi.get("memory.used") or 0)
    free_mib = float(smi.get("memory.free") or 0)
    total_gib = total_mib / 1024

    _head("THE REAL GPU  (driver + CUDA runtime - authoritative)")
    print(f"  Device            : {smi.get('name')}")
    print(f"  VRAM              : {total_gib:.1f} GiB total "
          f"({total_mib:,.0f} MiB)")
    print(f"  VRAM in use       : {used_mib / 1024:.1f} GiB   |   free: {free_mib / 1024:.1f} GiB")
    print(f"  Compute capability: {smi.get('compute_cap')}  (Blackwell = 12.0 / sm_120)")
    print(f"  Driver            : {smi.get('driver_version')}    VBIOS: {smi.get('vbios_version')}")
    cv = cuda_version()
    if cv:
        print(f"  CUDA runtime      : {cv}")
    print(f"  Power limit       : {smi.get('power.limit')} W   "
          f"(drawing {smi.get('power.draw')} W now)")
    print(f"  Temperature       : {smi.get('temperature.gpu')} C   "
          f"utilisation: {smi.get('utilization.gpu')} %")
    print(f"  Max SM clock      : {smi.get('clocks.max.sm')} MHz")
    print(f"  PCIe              : gen {smi.get('pcie.link.gen.max')} x{smi.get('pcie.link.width.max')}")
    vm = virtualization()
    if vm:
        print(f"  Virtualization    : {vm}")
    print(f"  Serial / UUID     : {smi.get('serial')}  /  {smi.get('uuid')}")

    tv = torch_view()
    if tv:
        _head("WHAT PYTORCH SEES  (what Sizzle actually renders through)")
        if not tv.get("cuda"):
            print(f"  torch {tv['torch']} is installed but CUDA is NOT available.")
            print("  Sizzle cannot render locally in this state - reinstall the CUDA wheels")
            print("  (see install.bat / requirements.txt).")
        else:
            print(f"  torch             : {tv['torch']}")
            print(f"  Device            : {tv['name']}")
            print(f"  Compute capability: {tv['cc']}    "
                  f"streaming multiprocessors: {tv['sms']}")
            print(f"  VRAM              : {tv['vram_bytes'] / 1024 ** 3:.1f} GiB "
                  f"({tv['vram_bytes']:,} bytes)")
            print(f"  bfloat16          : {'supported' if tv['bf16'] else 'NOT supported'}")
            archs = tv.get("arch_list", [])
            print(f"  Compiled for      : {', '.join(archs)}")
            if tv["cc"] in archs:
                print(f"  -> {tv['cc']} is compiled in: native kernels, no PTX JIT at startup.")
            else:
                print(f"  -> WARNING: {tv['cc']} is NOT in this build's arch list. Kernels will")
                print("     be JIT-compiled from PTX (slow first run) or fail outright.")

    win = windows_view()
    if win:
        _head("WHAT WINDOWS THINKS  (do not believe this part)")
        for row in win:
            ram = row["ram"]
            try:
                ram_gb = int(ram) / 1024 ** 3
                ram_s = f"{ram_gb:.2f} GB"
            except (TypeError, ValueError):
                ram_s = "(not reported)"
            print(f"  {row['name']}")
            print(f"      AdapterRAM {ram_s:>16s}   driver {row['driver']}")
        print()
        print("  Win32_VideoController.AdapterRAM is a 32-bit field, so any card with")
        print("  4 GiB or more wraps around and reports ~4.29 GB. Device Manager, dxdiag")
        print("  and Task Manager all read that field, which is why they undersell this")
        print("  card by an order of magnitude. The numbers above, from the driver and")
        print("  the CUDA runtime, are the real ones.")

    # ---- what it means for Sizzle -----------------------------------------
    _head("WHAT THIS MEANS FOR SIZZLE")
    # 22B params: bf16 = 2 bytes/param, fp8 = 1, nvfp4 ~= 0.5. Only ONE transformer
    # is resident at a time (each stage builds, denoises, then frees), plus the
    # Gemma text encoder and the VAEs, plus activations for the latent grid.
    weights_bf16 = 22e9 * 2 / 1024 ** 3
    weights_fp8 = 22e9 * 1 / 1024 ** 3
    print(f"  LTX-2.3 is a 22B model. One transformer is resident at a time:")
    print(f"     bf16 weights  ~{weights_bf16:.0f} GiB      fp8 weights  ~{weights_fp8:.0f} GiB")
    print(f"  This card has    {total_gib:.1f} GiB.")
    print()
    if total_gib >= 70:
        print("  -> Plenty of headroom. Run bf16 (SIZZLE_QUANTIZATION=none, the default)")
        print("     for the best quality, at any preset including 1920x1088.")
        print("     fp8-scaled-mm is still worth trying: on Blackwell it is FASTER.")
    elif total_gib >= 40:
        print("  -> bf16 fits, but without much room. Prefer the smaller presets, or")
        print("     set SIZZLE_QUANTIZATION=fp8-scaled-mm for headroom and speed.")
    elif total_gib >= 20:
        print("  -> Too tight for bf16. Set SIZZLE_QUANTIZATION=fp8-scaled-mm (or")
        print("     nvfp4-cast on Blackwell) and stick to the smaller presets.")
    else:
        print("  -> Not enough VRAM to hold the model. Use SIZZLE_OFFLOAD=cpu (needs")
        print("     ~36 GB of system RAM) or SIZZLE_OFFLOAD=disk, and expect it to be")
        print("     much slower. SIZZLE_BACKEND=mock renders placeholders with no GPU.")
    cc = (smi.get("compute_cap") or "").strip()
    if cc.startswith("12."):
        print()
        print("  -> Blackwell (sm_120) also unlocks NVFP4: SIZZLE_QUANTIZATION=nvfp4-cast")
        print("     is the smallest and fastest option if you want to trade some fidelity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
