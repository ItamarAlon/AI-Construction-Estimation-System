import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv:
    # Load environment variables from local .env file if present.
    load_dotenv()


def get_openrouter_api_key() -> str:
    """Fetch OPENROUTER_API_KEY from environment variables."""
    try:
        return os.environ["OPENROUTER_API_KEY"]
    except KeyError as exc:
        raise KeyError(
            "OPENROUTER_API_KEY is not set in environment variables."
        ) from exc


if __name__ == "__main__":
    api_key = get_openrouter_api_key()
    print(f"Loaded OpenRouter API key (length: {len(api_key)})")
