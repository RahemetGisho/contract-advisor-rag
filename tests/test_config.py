import pytest

from src.utils.config import ENV_OVERRIDES, load_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ENV_OVERRIDES:
        monkeypatch.delenv(var, raising=False)


def test_defaults_come_from_yaml():
    cfg = load_config()
    assert cfg["llm"]["provider"] == "groq"
    assert cfg["embeddings"]["provider"] == "sentence_transformers"


def test_env_var_overrides_a_single_value(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "mock")
    cfg = load_config()
    assert cfg["llm"]["provider"] == "mock"
    assert cfg["embeddings"]["provider"] == "sentence_transformers"  # untouched


def test_empty_env_var_is_ignored(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "")
    assert load_config()["llm"]["provider"] == "groq"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# ---- typed validation ---------------------------------------------------

import yaml  # noqa: E402

from src.utils.config import DEFAULT_CONFIG_PATH, ConfigError  # noqa: E402


def _write_variant(tmp_path, mutate):
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    mutate(cfg)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_shipped_config_is_valid_and_returns_a_plain_dict():
    cfg = load_config()
    assert isinstance(cfg, dict) and cfg["retrieval"]["top_k"] == 5


def test_misspelled_key_is_rejected_not_ignored(tmp_path):
    def typo(cfg):
        cfg["retrieval"]["top_K"] = cfg["retrieval"].pop("top_k")

    with pytest.raises(ConfigError) as exc:
        load_config(_write_variant(tmp_path, typo))
    assert "retrieval.top_k" in str(exc.value) or "retrieval.top_K" in str(exc.value)


def test_wrong_type_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="retrieval.top_k"):
        load_config(
            _write_variant(tmp_path, lambda c: c["retrieval"].update(top_k="five"))
        )


@pytest.mark.parametrize(
    "section,key,bad",
    [
        ("llm", "temperature", 9.0),
        ("llm", "timeout_seconds", 0),
        ("retrieval", "top_k", 0),
        ("data", "eval_holdout_fraction", 1.0),
        ("evaluation", "ragas_eval_set_size", 0),
        ("api", "rate_limit_per_minute", 0),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, section, key, bad):
    with pytest.raises(ConfigError, match=f"{section}.{key}"):
        load_config(_write_variant(tmp_path, lambda c: c[section].update({key: bad})))


def test_overlap_must_be_smaller_than_chunk_size(tmp_path):
    def bad(cfg):
        cfg["chunking"]["chunk_overlap_chars"] = cfg["chunking"]["chunk_size_chars"]

    with pytest.raises(ConfigError, match="overlap"):
        load_config(_write_variant(tmp_path, bad))


def test_unknown_provider_is_caught_at_load_time(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "nonsense")
    with pytest.raises(ConfigError, match="llm.provider"):
        load_config()


def test_require_key_env_override_coerces_a_string_to_bool(monkeypatch):
    monkeypatch.setenv("RAG_API_REQUIRE_KEY", "false")
    assert load_config()["api"]["require_api_key"] is False


# ---- .env loading -----------------------------------------------------------

import os  # noqa: E402

from src.utils import config as config_module  # noqa: E402


def _clean(monkeypatch, var):
    """Register restore-to-absent so a test cannot leak an env var."""
    monkeypatch.setenv(var, "placeholder")
    monkeypatch.delenv(var)


def test_env_file_fills_variables_that_are_not_already_set(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=from-the-file\nAPI_KEYS=" + "a" * 20 + "\n")
    monkeypatch.setattr(config_module, "ENV_PATH", env_file)
    _clean(monkeypatch, "GROQ_API_KEY")
    _clean(monkeypatch, "API_KEYS")
    config_module.load_env()
    assert os.environ["GROQ_API_KEY"] == "from-the-file"
    assert os.environ["API_KEYS"] == "a" * 20


def test_real_environment_wins_over_the_env_file(tmp_path, monkeypatch):
    """Docker and CI inject real variables; a stale local .env must never override them."""
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=from-the-file\n")
    monkeypatch.setattr(config_module, "ENV_PATH", env_file)
    monkeypatch.setenv("GROQ_API_KEY", "from-the-real-environment")
    config_module.load_env()
    assert os.environ["GROQ_API_KEY"] == "from-the-real-environment"


def test_missing_env_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / "does-not-exist")
    config_module.load_env()  # must simply do nothing
