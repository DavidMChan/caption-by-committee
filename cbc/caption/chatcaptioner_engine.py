import logging
from typing import List, Optional, Union

from PIL.Image import Image
from tqdm import tqdm

from cbc.caption.base import CaptionEngine
from cbc.lm import LMEngine

QUESTION_INSTRUCTION = (
    "I have an image. "
    "Ask me questions about the content of this image. "
    "Carefully asking me informative questions to maximize your information about this image content. "
    "Each time ask one question only without giving an answer. "
    "Avoid asking yes/no questions."
    'I\'ll put my answer beginning with "Answer:".'
)

SUB_QUESTION_INSTRUCTION = "Next Question. Avoid asking yes/no questions. \n" "Question: "


SUMMARY_INSTRUCTION = (
    "Now summarize the information you get in a few sentences. "
    "Ignore the questions with answers no or not sure. "
    "Don't add information. Don't miss information. \n"
    "Summary: "
)


ANSWER_INSTRUCTION = "Answer given questions. If you are not sure about the answer, say you don't know honestly. Don't imagine any contents that are not in the image."


SUB_ANSWER_INSTRUCTION = "Answer: "  # template following blip2 huggingface demo


FIRST_QUESTION = "Describe this image in detail."


VALID_CHATGPT_MODELS = ["gpt-3.5-turbo"]
VALID_GPT3_MODELS = ["text-davinci-003", "text-davinci-002", "davinci"]


def get_chat_log(questions, answers, last_n=-1):
    n_addition_q = len(questions) - len(answers)
    assert (n_addition_q) in [0, 1]
    template = "Question: {} \nAnswer: {} \n"
    chat_log = ""
    if last_n > 0:
        answers = answers[-last_n:]
        questions = questions[-(last_n + n_addition_q) :]
    elif last_n == 0:
        answers = []
        questions = questions[-1:] if n_addition_q else []

    for i in range(len(answers)):
        chat_log = chat_log + template.format(questions[i], answers[i])
    return (
        f"{chat_log}Question: {questions[-1]}"
        if n_addition_q
        else chat_log[:-2]
    )


def prepare_gpt_prompt(task_prompt, questions, answers, sub_prompt):
    return "\n".join([task_prompt, get_chat_log(questions, answers), sub_prompt])


class AskQuestions:
    def __init__(self, img: Image, blip2: CaptionEngine, lm_engine: LMEngine, n_blip2_context=-1):
        self.img = img
        self.blip2 = blip2
        self._lm_engine = lm_engine
        self.n_blip2_context = n_blip2_context

        self.questions = []
        self.answers = []

    def reset(self, img):
        self.img = img
        self.questions = []
        self.answers = []

    def ask_question(self):
        if len(self.questions) == 0:
            # first question is given by human to request a general discription
            return FIRST_QUESTION
        gpt_prompt = prepare_gpt_prompt(
            QUESTION_INSTRUCTION, self.questions, self.answers, SUB_QUESTION_INSTRUCTION
        )
        return self._lm_engine.best(gpt_prompt)

    def question_trim(self, question):
        question = question.split("Question: ")[-1].replace("\n", " ").strip()
        if "Answer:" in question:  # Some models make up an answer after asking. remove it
            q, a = question.split("Answer:")[:2]
            question = a.strip() if len(q) == 0 else q.strip()
        return question

    def answer_question(self):
        # prepare the context for blip2
        blip2_prompt = "\n".join(
            [
                ANSWER_INSTRUCTION,
                # get_chat_log(self.questions, self.answers, last_n=self.n_blip2_context),
                f"Question: {self.questions[-1]}",
                SUB_ANSWER_INSTRUCTION,
            ]
        )

        return self.blip2.get_ask_caption(self.img, blip2_prompt)

    def answer_trim(self, answer):
        answer = answer.split("Question:")[0].replace("\n", " ").strip()
        return answer

    def chatting(self, n_rounds, print_mode):
        logging.debug("--------Chat Starts----------")

        for _ in tqdm(range(n_rounds), desc="Chat Rounds", disable=print_mode != "bar"):
            question = self.ask_question()
            question = self.question_trim(question)
            self.questions.append(question)

            logging.debug(f"LM_Egine: {question}")

            answer = self.answer_question()
            answer = self.answer_trim(answer)
            self.answers.append(answer)

            logging.debug(f"BLIP-2: {answer}")

        logging.debug("--------Chat Ends----------")

        return self.questions, self.answers


def summarize_chat(questions, answers, _lm_engine):
    summary_prompt = prepare_gpt_prompt(QUESTION_INSTRUCTION, questions, answers, SUMMARY_INSTRUCTION)
    summary = _lm_engine.best(summary_prompt)

    summary = summary.replace("\n", " ").strip()
    return summary, summary_prompt


class ChatCaptionerEngine(CaptionEngine):
    def __init__(
        self,
        language_model: Union[LMEngine, str] = "gpt3_davinci3",
        caption_engine: Union[CaptionEngine, str] = "blip2-t5-xl",
        device: Optional[str] = None,
    ):
        # BlipCaption
        if isinstance(caption_engine, str):
            from cbc.caption import CAPTION_ENGINES_CLI

            self.captioner = CAPTION_ENGINES_CLI[caption_engine](device="cuda")
        else:
            self.captioner = caption_engine

        # Setup the language model
        self._language_model = (
            language_model if isinstance(language_model, LMEngine) else LMEngine.from_string(language_model)
        )
        self._device = self.captioner._device

    def __call__(self, raw_image: Image, n_captions: int = 1, temperature: Optional[float] = None) -> List[str]:
        if n_captions == 1:
            return [self.get_baseline_caption(raw_image)]
        return [self.get_baseline_caption(raw_image) for _ in range(n_captions)]

    def get_baseline_caption(self, image: Image, n_rounds: int = 10, n_blip2_context: int = -1, print_mode: str = "no"):
        chat = AskQuestions(image, self.captioner, n_blip2_context=n_blip2_context, lm_engine=self._language_model)
        questions, answers = chat.chatting(n_rounds, print_mode=print_mode)
        summary, summary_prompt = summarize_chat(questions, answers, self._language_model)
        logging.debug(f"Summary Prompt: {summary_prompt}")
        logging.debug(f"Summary: {summary}")
        results = {
            "ChatCaptioner": {"caption": summary, "chat": summary_prompt},
            "BLIP2+OurPrompt": {"caption": answers[0]},
        }
        # Default BLIP2 caption
        caption = self.captioner.get_baseline_caption(image)
        results["BLIP2"] = {"caption": caption}

        return results["ChatCaptioner"]["caption"]
