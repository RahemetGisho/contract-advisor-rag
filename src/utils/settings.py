"""Typed schema for configs/config.yaml.

load_config() validates the YAML against this schema, so a misspelled
key, a wrong type, or an out-of-range value fails at startup with a
message naming the exact field, instead of surfacing later as a
KeyError deep inside a request. Unknown keys are rejected (extra="forbid"),
which is what catches typos like `top_K`.

The validated result is returned as a plain dict, so code that reads
config["llm"]["provider"] is unchanged.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    # extra="forbid" rejects misspelled keys. protected_namespaces=() is needed
    # because config fields legitimately start with "model_" (model_name), which
    # Pydantic otherwise reserves and warns about on every import.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ProjectCfg(_Section):
    name: str
    random_seed: int


class DataCfg(_Section):
    raw_dir: str
    processed_dir: str
    cuad_qa_path: str
    eval_holdout_fraction: float = Field(gt=0.0, lt=1.0)


class ChunkingCfg(_Section):
    strategy: Literal["clause_aware", "fixed_size"]
    chunk_size_chars: int = Field(gt=0)
    chunk_overlap_chars: int = Field(ge=0)

    @model_validator(mode="after")
    def overlap_smaller_than_chunk(self):
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError(
                "chunk_overlap_chars must be smaller than chunk_size_chars"
            )
        return self


class EmbeddingsCfg(_Section):
    provider: Literal["sentence_transformers", "tfidf"]
    model_name: str
    dimension: int = Field(gt=0)
    tfidf_dimension: int = Field(gt=0)


class VectorStoreCfg(_Section):
    provider: Literal["chroma"]
    persist_dir: str
    collection_name: str = Field(min_length=1)


class RetrievalCfg(_Section):
    top_k: int = Field(ge=1)
    score_threshold: float = Field(ge=0.0, le=1.0)


class LlmCfg(_Section):
    provider: Literal["groq", "mock"]
    model_name: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0.0)
    max_retries: int = Field(ge=0)


class GenerationCfg(_Section):
    min_context_chunks: int = Field(ge=0)
    abstain_message: str = Field(min_length=1)


class QualityGateCfg(_Section):
    min_faithfulness: float = Field(ge=0.0, le=1.0)
    min_answer_relevancy: float = Field(ge=0.0, le=1.0)
    min_context_precision: float = Field(ge=0.0, le=1.0)
    min_context_recall: float = Field(ge=0.0, le=1.0)


class RetrievalGateCfg(_Section):
    min_recall_at_k: float = Field(ge=0.0, le=1.0)
    min_mrr: float = Field(ge=0.0, le=1.0)


class EvaluationCfg(_Section):
    ragas_eval_set_size: int = Field(ge=1)
    quality_gate: QualityGateCfg
    retrieval_quality_gate: RetrievalGateCfg


class ApiCfg(_Section):
    query_log_path: str
    require_api_key: bool
    rate_limit_per_minute: int = Field(ge=1)


class LoggingCfg(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(_Section):
    project: ProjectCfg
    data: DataCfg
    chunking: ChunkingCfg
    embeddings: EmbeddingsCfg
    vector_store: VectorStoreCfg
    retrieval: RetrievalCfg
    llm: LlmCfg
    generation: GenerationCfg
    evaluation: EvaluationCfg
    api: ApiCfg
    logging: LoggingCfg
