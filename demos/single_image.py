import streamlit as st
import torch
from PIL import Image

from cbc.caption import CAPTION_ENGINES, CaptionEngine
from cbc.caption.utils import postprocess_caption
from cbc.caption_by_committee import DEFAULT_CBC_PROMPT, get_prompt_for_candidates
from cbc.lm import LM_ENGINES, LM_LOCAL_ENGINES, LMEngine


@st.cache(allow_output_mutation=True)
def _get_caption_engine(engine_name: str) -> CaptionEngine:
    return CAPTION_ENGINES[engine_name](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore


@st.cache(allow_output_mutation=True)
def _get_lm_engine(engine_name: str) -> LMEngine:
    if engine_name in LM_LOCAL_ENGINES:
        return LM_ENGINES[engine_name](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    return LM_ENGINES[engine_name]()


# Streamlit setup
st.set_page_config(
    page_title="IC3: Image Captioning by Committee Consensus",
    page_icon="📸",
    layout="centered",
)

# Streamlit title
st.title("IC3: Image Captioning by Committee Consensus")

# Streamlit description
st.markdown(
    """
    This is a demo of the paper IC3: Image Captioning by Committee Consensus.
    """
)

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

with st.expander("Model Hyperparameters (Advanced)"):
    # Selection for sampling parameters
    st.subheader("Image Captioning Model")

    # Selection box for image captioning model
    model = st.selectbox(
        "Image Captioning Engine",
        CAPTION_ENGINES.keys(),
    )

    # Two columns
    col1, col2 = st.columns(2)

    with col1:
        # Number of samples
        num_samples = st.number_input(
            "Number of samples",
            min_value=1,
            max_value=25,
            value=20,
            step=1,
        )

    with col2:
        # Temperature
        temperature = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=1.05,
            step=0.05,
        )

    st.subheader("Summarization Model")

    # Selection box for summarization engine
    engine = st.selectbox(
        "Select a summarization engine",
        LM_ENGINES.keys(),
    )

    # engine_temperature = st.number_input("Temperature", min_value=0.1, max_value=1.0, value=0.9, step=0.05, key=1)

    prompt = st.text_area(
        "Prompt, use {} to indicate where the image captions should be inserted",
        DEFAULT_CBC_PROMPT,
        height=125,
    )


# Streamlit button
if st.button("Generate Caption"):
    if uploaded_file is None:
        st.error("Please upload an image first!")
        st.stop()

    # Get the caption engine
    captioner = _get_caption_engine(model)  # type: ignore
    sm_captioner = _get_caption_engine("Socratic Models")  # type: ignore

    # Get the summarization engine
    summarizer = _get_lm_engine(engine)  # type: ignore

    with st.spinner("Generating caption..."):
        raw_image = Image.open(uploaded_file).convert("RGB")
        captions = captioner(raw_image, n_captions=int(num_samples), temperature=temperature)
        baseline_caption = captions[0]
        sm_baseline = postprocess_caption(sm_captioner.get_baseline_caption(raw_image))
        output_prompt = get_prompt_for_candidates(captions, prompt=prompt)
        summary = postprocess_caption(summarizer.best(output_prompt))

    # Step 3: Display the results
    st.header("Results")

    # Display the image
    st.image(uploaded_file, use_column_width=True)

    # Display the caption
    st.caption(f"IC3 Caption ({model} + {engine}): {summary}")
    st.caption(f"Compare to {model} (baseline, 16 beams): {baseline_caption}")
    st.caption(f"Compare to Socratic Models (baseline): {sm_baseline}")

    # Display the sampled captions
    st.subheader("Generated Samples (Which were summarized)")
    st.write(captions)

    # Display the prompt
    st.subheader("Prompt")
    st.write(output_prompt)
