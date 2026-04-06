from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PromptCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    quantization: str = Field(min_length=1)
    model_id: str | None = None
    model_path: str | None = None


class RunResult(BaseModel):
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
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Compatibility helper for existing call sites expecting dict output."""
        return self.model_dump()
