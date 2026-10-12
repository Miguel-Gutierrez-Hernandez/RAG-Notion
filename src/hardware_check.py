"""Detección de hardware: CPU, RAM y GPU (NVIDIA, AMD, Apple Silicon)."""
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

import psutil


@dataclass
class GPUInfo:
    vendor: str            # "nvidia" | "amd" | "apple" | "none"
    name: str = ""
    vram_gb: float = 0.0   # en Apple Silicon es memoria unificada


@dataclass
class HardwareInfo:
    os: str
    cpu: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu: GPUInfo = field(default_factory=lambda: GPUInfo("none"))

    @property
    def has_gpu(self) -> bool:
        return self.gpu.vendor != "none"


def _detect_nvidia() -> GPUInfo | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
        if not out:
            return None
        name, mem_mb = [p.strip() for p in out[0].split(",")]
        return GPUInfo("nvidia", name, round(float(mem_mb) / 1024, 1))
    except (subprocess.SubprocessError, ValueError):
        return None


def _detect_amd() -> GPUInfo | None:
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        vram_gb = 0.0
        for line in out.splitlines():
            if "VRAM Total Memory (B)" in line:
                vram_gb = round(int(line.split(":")[-1].strip()) / 1024**3, 1)
                break
        return GPUInfo("amd", "AMD GPU (ROCm)", vram_gb)
    except (subprocess.SubprocessError, ValueError):
        return None


def _detect_apple() -> GPUInfo | None:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        ram = round(psutil.virtual_memory().total / 1024**3, 1)
        return GPUInfo("apple", "Apple Silicon (memoria unificada)", ram)
    return None


def detect_gpu() -> GPUInfo:
    for detector in (_detect_nvidia, _detect_amd, _detect_apple):
        gpu = detector()
        if gpu:
            return gpu
    return GPUInfo("none")


def detect_hardware() -> HardwareInfo:
    vm = psutil.virtual_memory()
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or platform.machine(),
        cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        ram_total_gb=round(vm.total / 1024**3, 1),
        ram_available_gb=round(vm.available / 1024**3, 1),
        gpu=detect_gpu(),
    )
