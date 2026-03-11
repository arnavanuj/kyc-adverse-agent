from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "KYC Adverse Media Agent"
    app_env: str = "dev"
    db_path: str = "./data/kyc_memory.db"
    max_articles: int = 20
    search_results_per_query: int = 8
    reflection_max_loops: int = 1
    request_timeout_seconds: int = 15

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
