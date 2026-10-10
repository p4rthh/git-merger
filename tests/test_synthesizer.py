from git_merger.model_router import ModelRouter
from git_merger.synthesizer import SemanticSynthesizer

if __name__ == "__main__":
    router = ModelRouter()
    synthesizer = SemanticSynthesizer(router.client)
    print("Synthesizer initialized:", synthesizer.model)