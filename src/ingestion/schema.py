from pydantic import BaseModel, Field, field_validator


class ContractQARecord(BaseModel):
    contract_id: str = Field(..., min_length=1)
    contract_title: str = Field(..., min_length=1)
    clause_category: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)
    answer_texts: list[str] = Field(default_factory=list)
    answer_starts: list[int] = Field(default_factory=list)
    is_impossible: bool = Field(default=False)

    @field_validator("answer_starts")
    @classmethod
    def answer_starts_non_negative(cls, v: list[int]) -> list[int]:
        if any(start < 0 for start in v):
            raise ValueError("every answer_start must be non-negative")
        return v

    @field_validator("answer_starts")
    @classmethod
    def starts_match_texts_count(cls, v: list[int], info) -> list[int]:
        texts = info.data.get("answer_texts", [])
        if len(v) != len(texts):
            raise ValueError(
                f"answer_starts has {len(v)} entries but answer_texts has {len(texts)} -- "
                "every answer span needs both its text and its offset"
            )
        return v
