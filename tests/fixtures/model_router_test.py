from dotenv import load_dotenv

from git_merger.model_router import ModelRouter

if __name__ == "__main__":
    load_dotenv()
    router = ModelRouter()

    # Check model selection without making an API request.
    route = router.route("intent_synthesis")

    print("Router initialized successfully!")
    print("Model:", route.model_id)
    print("Max tokens:", route.max_tokens)
    print("Temperature:", route.temperature)
    print("JSON mode:", route.json_mode)

    result = router.call(
        route=router.route("strategy_selection"),
        system_prompt="Return a JSON object with a key named status.",
        user_prompt='Set status to "working".',
    )

    print(result)