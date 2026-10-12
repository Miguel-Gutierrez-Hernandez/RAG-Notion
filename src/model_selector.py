"""Elige el modelo de chat según el hardware detectado."""
from dataclasses import dataclass

from src.hardware_check import HardwareInfo


@dataclass
class ModelChoice:
    chat_model: str
    tier: str
    reason: str


def select_chat_model(hw: HardwareInfo, override: str = "") -> ModelChoice:
    if override:
        return ModelChoice(override, "manual", "Definido en CHAT_MODEL (.env)")

    gpu = hw.gpu
    if gpu.vendor in ("nvidia", "amd") and gpu.vram_gb >= 12:
        return ModelChoice("qwen2.5:14b", "gpu-alta",
                           f"GPU con {gpu.vram_gb} GB de VRAM")
    if gpu.vendor in ("nvidia", "amd") and gpu.vram_gb >= 6:
        return ModelChoice("qwen2.5:7b", "gpu-media",
                           f"GPU con {gpu.vram_gb} GB de VRAM")
    if gpu.vendor == "apple" and gpu.vram_gb >= 16:
        return ModelChoice("qwen2.5:7b", "apple-silicon",
                           f"Memoria unificada de {gpu.vram_gb} GB")
    if hw.ram_total_gb >= 12:
        return ModelChoice("qwen2.5:3b", "cpu",
                           "Solo CPU: modelo ligero para respuestas ágiles "
                           "(puedes probar qwen2.5:7b en CHAT_MODEL)")
    return ModelChoice("qwen2.5:1.5b", "cpu-limitada",
                       f"Poca RAM disponible ({hw.ram_available_gb} GB)")
