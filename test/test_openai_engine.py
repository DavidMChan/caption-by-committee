from cbc.lm.openai_engine import OpenAILMEngine

prompt = """We aim for building a unified multimodal multitask AI system. Toward this goal,"""


def test_hf_engine() -> None:
    engine = OpenAILMEngine(model="text-davinci-002")
    completions = engine(prompt, n_completions=5)
    for completion in completions:
        print(completion)
        print("")
        print("-----------")
        print("")


if __name__ == "__main__":
    test_hf_engine()
