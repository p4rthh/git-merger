from git_merger.model_router import ModelRouter
from git_merger.synthesizer import SemanticSynthesizer

router = ModelRouter()
synthesizer = SemanticSynthesizer(router.client)

print("Synthesizer initialized:", synthesizer.model)