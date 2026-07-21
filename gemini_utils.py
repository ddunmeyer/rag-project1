# gemini_utils.py
# ---------------
# Shared helpers for calling Gemini with automatic retry on rate limits.

import re
import time

from google.genai import types

from config import GEMINI_MAX_RETRIES


def _parse_retry_seconds(error):
    """Read Gemini's suggested retry delay from a 429 error message."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1
    return 15


def call_gemini(client, *, model, contents, temperature=0.2):
    """
    Call Gemini and retry automatically when the free-tier rate limit is hit.

    Returns the generate_content response object.
    """
    config = types.GenerateContentConfig(temperature=temperature)

    last_error = None
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as error:
            last_error = error
            error_text = str(error).lower()
            is_rate_limit = (
                "429" in error_text
                or "resource_exhausted" in error_text
                or "rate limit" in error_text
                or "quota" in error_text
            )
            if is_rate_limit and attempt < GEMINI_MAX_RETRIES - 1:
                time.sleep(_parse_retry_seconds(error))
                continue
            raise

    raise last_error
