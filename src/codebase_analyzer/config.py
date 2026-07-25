"""Runtime configuration loaded from environment variables."""

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnalyzerSettings(BaseSettings):
    """Configuration shared by tokenization and LLM pipeline components."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CODE_ANALYZER_",
        extra="ignore",
    )

    routine_model: str = "gpt-5.6-luna"
    complex_model: str = "gpt-5.6-terra"
    synthesis_model: str = "gpt-5.6-sol"
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    max_context_tokens: int = Field(default=32_768, gt=0)
    reserved_output_tokens: int = Field(default=4_096, gt=0)
    chunk_token_budget: int = Field(default=12_000, gt=0)
    max_concurrency: int = Field(default=4, ge=1, le=32)
    request_timeout_seconds: float = Field(default=45.0, gt=0, le=300)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_token_budgets(self) -> "AnalyzerSettings":
        prompt_budget = self.max_context_tokens - self.reserved_output_tokens
        if prompt_budget <= 0:
            raise ValueError("reserved output tokens must be below the context limit")
        if self.chunk_token_budget > prompt_budget:
            raise ValueError("chunk token budget exceeds available prompt tokens")
        return self

    @property
    def prompt_token_budget(self) -> int:
        """Maximum tokens available for prompts, including instructions."""

        return self.max_context_tokens - self.reserved_output_tokens
