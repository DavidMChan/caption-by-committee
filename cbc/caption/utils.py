def postprocess_caption(caption: str, method: str = "all") -> str:
    caption = caption.strip()

    # Remove any floating special tokens, make sure there's a . at the end, and capitalize the first letter

    caption = caption.replace("<s>", "").replace("</s>", "")
    caption = caption.replace("<pad>", "").replace("</pad>", "")

    # Replace \n with a space
    caption = caption.replace("\n", " ")

    # Replace multiple spaces with a single space
    caption = " ".join(caption.split())

    # Make sure there's a . at the end
    if not caption.endswith("."):
        caption += "."

    # If there are spaces around a period, remove them
    caption = caption.replace(" .", ".")

    # Remove any case variants of "I'm not sure, but the image is likely of"
    if caption.lower().startswith("i'm not sure, but the image is likely of"):
        caption = caption[len("I'm not sure, but the image is likely of") :]
    caption = caption.strip()

    # Make sure the first letter in each sentence is capitalized
    if method != "no_caps":
        caption = ". ".join([s[0].upper() + s[1:] for s in caption.split(". ")])

    # Remove any double periods
    caption = caption.replace("..", ".")

    if "truncate" in method:
        # Truncate the caption to the first period
        if "." in caption:
            caption = caption[: caption.index(".") + 1]

    # Clean up the word 'Summary: ' if it exists
    caption = caption.replace("Summary: ", "").replace("summary:", "")

    caption = caption.strip()

    # Handle the case where you have some content, then "A: " or "A." or a sentence with a question ? followed by an "A:"
    if "A." in caption or "A:" in caption or "Answer:" in caption or "Comment:" in caption:
        if "?" in caption:
            caption = caption[: caption.index("?") + 1]
            # Split the caption into sentences
            sentences = caption.split(".")
            sentences = [s.strip() for s in sentences if s.strip()]
            caption = ". ".join(sentences[:-1])
        else:
            # Remove everything after the "A./:"
            if "A." in caption:
                caption = caption[: caption.index("A.")]
            elif "A:" in caption:
                caption = caption[: caption.index("A:")]
            elif "Answer:" in caption:
                caption = caption[: caption.index("Answer:")]

            if "Comment:" in caption:
                caption = caption[: caption.index("Comment:")]

    caption = caption.strip()

    # If any sentence repeats, remove it
    sentences = caption.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    output_sentences = []
    for s in sentences:
        if s not in output_sentences:
            output_sentences.append(s)
    caption = ". ".join(output_sentences)

    caption = caption.strip()

    # Make sure there's a . at the end
    if not caption.endswith("."):
        caption += "."

    return caption.strip()
