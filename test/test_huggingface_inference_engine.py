from cbc.lm.huggingface_inference_engine import HuggingfaceInferenceLMEngine


def test_hf_engine() -> None:
    engine = HuggingfaceInferenceLMEngine(model="gpt2")
    completions = engine("The quick brown fox is", n_completions=5)
    for completion in completions:
        print(completion)


if __name__ == "__main__":
    test_hf_engine()
