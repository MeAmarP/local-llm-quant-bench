from __future__ import annotations

"""Pydantic schema models used across quantbench pipelines."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PromptCase(BaseModel):
    """One benchmark prompt record loaded from JSONL prompt files."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class RunSpec(BaseModel):
    """Configuration for one benchmark run variant (backend + quantization + model reference)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    quantization: str = Field(min_length=1)
    model_id: str | None = None
    model_path: str | None = None


class RunResult(BaseModel):
    """Result payload for a single prompt execution under one run spec."""

    model_config = ConfigDict(extra="allow")

    run_name: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    quantization: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    model_ref: str = Field(min_length=1)
    prompt_chars: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_sec: float = Field(gt=0)
    tokens_per_sec: float | None = Field(default=None, ge=0)
    load_time_sec: float | None = Field(default=None, ge=0)
    peak_gpu_mem_mb: float | None = Field(default=None, ge=0)
    generated_text: str
    error: str | None = None
    # Extended performance metrics (feature/extended-metrics)
    ttft_ms: float | None = Field(default=None, ge=0)
    peak_ram_mb: float | None = Field(default=None, ge=0)
    avg_power_w: float | None = Field(default=None, ge=0)
    energy_per_token_j: float | None = Field(default=None, ge=0)
    # Quality evaluation fields (populated when golden answers are provided)
    quality_pass: bool | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_method: str | None = None
    quality_details: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Compatibility helper for existing call sites expecting dict output."""
        return self.model_dump()
