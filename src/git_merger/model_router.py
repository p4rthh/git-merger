
import json
import os
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class ModelRoute:
    model_id: str
    max_tokens: int
    temperature: float
    json_mode: bool


MODEL_ROUTES = {
    "intent_synthesis": ModelRoute(
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        max_tokens=8192,
        temperature=0.1,
        json_mode=True,
    ),
    "harness_generation": ModelRoute(
        model_id="nvidia/nemotron-3-super-120b-a12b",
        max_tokens=8192,
        temperature=0.2,
        json_mode=True,
    ),
    "log_parsing": ModelRoute(
        model_id="nvidia/nemotron-3-nano-30b-a3b",
        max_tokens=4096,
        temperature=0.0,
        json_mode=True,
    ),
    "repair": ModelRoute(
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        max_tokens=8192,
        temperature=0.15,
        json_mode=True,
    ),
    "strategy_selection": ModelRoute(
        model_id="nvidia/nemotron-3-nano-30b-a3b",
        max_tokens=2048,
        temperature=0.0,
        json_mode=True,
    ),
}


class ModelRouter:
    def __init__(self, api_key: str | None = None):
        """
        Initialize the Nebius API client.

        Reads NEBIUS_API_KEY from the environment unless api_key
        is explicitly provided.
        """
        key = api_key or os.getenv("NEBIUS_API_KEY")

        if not key:
            raise ValueError(
                "NEBIUS_API_KEY is missing. Add it to your .env file "
                "or set it as an environment variable."
            )

        self.client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1",
            api_key=key,
        )

    def route(self, task_type: str) -> ModelRoute:
        """Return the model configuration for a task."""
        if task_type not in MODEL_ROUTES:
            available = ", ".join(MODEL_ROUTES.keys())
            raise ValueError(
                f"Unknown task type: {task_type!r}. "
                f"Available task types: {available}"
            )

        return MODEL_ROUTES[task_type]

    def call(
        self,
        route: ModelRoute,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """
        Call the selected model.

        Retries failed API calls up to 3 times, with exponential
        backoff delays of 1, 2, and 4 seconds.
        """
        max_retries = 3

        for attempt in range(max_retries + 1):
            try:
                request_args = {
                    "model": route.model_id,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    "temperature": route.temperature,
                    "max_tokens": route.max_tokens,
                }

                if route.json_mode:
                    request_args["response_format"] = {
                        "type": "json_object"
                    }

                response = self.client.chat.completions.create(
                    **request_args
                )

                content = response.choices[0].message.content

                if content is None:
                    raise ValueError(
                        "The model returned an empty response."
                    )

                if route.json_mode:
                    return json.loads(content)

                return {"content": content}

            except Exception:
                if attempt == max_retries:
                    raise

                time.sleep(2 ** attempt)

        # Defensive fallback; normally unreachable.
        raise RuntimeError("Model call failed unexpectedly.")

    from git_merger.model_router import ModelRouter

router = ModelRouter()

route = router.route("intent_synthesis")

result = router.call(
    route=route,
    system_prompt="You are a Python merge-conflict expert.",
    user_prompt="Return a JSON object with a message field.",
)

print(result)