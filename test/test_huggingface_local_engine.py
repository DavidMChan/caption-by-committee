from cbc.lm.huggingface_local_engine import HuggingFaceLocalLMEngine, HuggingFaceLocalSummaryEngine


def test_hf_engine() -> None:
    engine = HuggingFaceLocalLMEngine(model="gpt2", device="cuda:1")
    completions = engine("The quick brown fox is", n_completions=5)
    for completion in completions:
        print(completion)


def test_hf_summary_engine() -> None:
    engine = HuggingFaceLocalSummaryEngine(model="t5-small", device="cuda:1")
    summary = engine.best("The quick brown fox is")
    print(summary)


if __name__ == "__main__":
    test_hf_engine()
    test_hf_summary_engine()
