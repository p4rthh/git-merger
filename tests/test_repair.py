
from git_merger.model_router import ModelRouter
from git_merger.repair import RepairSynthesizer

router = ModelRouter()
repair_synthesizer = RepairSynthesizer(router.client)

print("Repair synthesizer initialized.")
print("Model:", repair_synthesizer.model)
