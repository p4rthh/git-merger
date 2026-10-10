
from git_merger.model_router import ModelRouter
from git_merger.harness import HarnessGenerator

router = ModelRouter()
harness_generator = HarnessGenerator(router.client)

print("Harness generator initialized.")
print("Model:", harness_generator.synthesis_model)

