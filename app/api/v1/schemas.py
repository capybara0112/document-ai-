from typing import Any, Literal
from pydantic import BaseModel, Field


TaskType = Literal["AUTO", "INVOICE", "FINANCIAL_REPORT", "GENERAL"]


class AnalyzeResponse(BaseModel):
    status: Literal["SUCCESS", "PARTIAL", "FAILED"]
    file_type: str
    task_type: str
    processing_time_ms: int = Field(ge=0)
    pages_count: int | None = None
    extracted_data: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
