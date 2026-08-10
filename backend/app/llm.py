import os
import time
import logging
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("har_app")

_last_call_timestamp: float = 0.0
COOLDOWN_SECONDS: float = 1.0


@dataclass
class LLMResult:
    ok: bool
    text: Optional[str] = None
    error_code: Optional[str] = None  # "QUOTA_EXCEEDED" | "API_ERROR" | "NO_KEY" | "UNKNOWN"
    error_message: Optional[str] = None


SYSTEM_INSTRUCTIONS = (
    "You are an assistant that explains human activity recognition (HAR) model "
    "outputs. You must only use the structured data provided to you. Never invent "
    "sensor readings, timestamps, or activities that are not present in the input. "
    "Be concise, objective, and factual."
)


def _enforce_cooldown():
    global _last_call_timestamp
    now = time.time()
    elapsed = now - _last_call_timestamp
    if elapsed < COOLDOWN_SECONDS:
        sleep_dur = COOLDOWN_SECONDS - elapsed
        time.sleep(sleep_dur)
    _last_call_timestamp = time.time()


def _call_groq(prompt: str, retries: int = 1) -> LLMResult:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return LLMResult(
            ok=False,
            error_code="NO_KEY",
            error_message="GROQ_API_KEY is not set. Add it to your .env file."
        )

    try:
        from groq import Groq, RateLimitError, APIError
        client = Groq(api_key=api_key)
    except Exception as e:
        return LLMResult(ok=False, error_code="UNKNOWN", error_message=str(e))

    candidate_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]

    last_error_msg = None

    for attempt in range(retries + 1):
        _enforce_cooldown()

        for model_name in candidate_models:
            try:
                logger.info(f"[Groq LLM] Requesting model: {model_name} (Attempt {attempt})")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=250,
                    temperature=0.3,
                )
                if response and response.choices and response.choices[0].message.content:
                    output_text = response.choices[0].message.content.strip()
                    logger.info(f"[Groq LLM] Success from model: {model_name}")
                    return LLMResult(ok=True, text=output_text)
            except RateLimitError as e:
                logger.error(f"[Groq LLM] Rate limit hit on {model_name}: {e}")
                if attempt < retries:
                    time.sleep(1.5)
                    break
                return LLMResult(
                    ok=False,
                    error_code="QUOTA_EXCEEDED",
                    error_message=(
                        "AI summary temporarily unavailable — Groq usage limit reached. "
                        "Try again in a moment or check your API key quota."
                    ),
                )
            except APIError as e:
                logger.warning(f"[Groq LLM] API error on {model_name}: {e}")
                last_error_msg = str(e)
            except Exception as e:
                logger.warning(f"[Groq LLM] Error on {model_name}: {type(e).__name__}: {e}")
                last_error_msg = str(e)

        if attempt < retries:
            time.sleep(1.0 * (attempt + 1))

    return LLMResult(
        ok=False,
        error_code="API_ERROR",
        error_message=f"Groq API error: {last_error_msg or 'Request failed'}"
    )


def explain_prediction(payload: dict) -> LLMResult:
    prompt = (
        f"Prediction payload:\n{payload}\n\n"
        "Explain this prediction in 2-3 concise sentences. Mention top activity, "
        "confidence percentage, whether it is a clear landslide or close result, and notable biomechanical motion traits."
    )
    return _call_groq(prompt)


def generate_explanation(payload: dict) -> LLMResult:
    return explain_prediction(payload)
