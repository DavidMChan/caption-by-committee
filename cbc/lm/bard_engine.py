"""
Some of the code below is based on https://github.com/acheong08/Bard
Copyright (c) 2023 Antonio Cheong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import json
import logging
import os
import random
import re
import string
import time
from typing import Any, List, Optional

import requests
from ratelimit import limits, sleep_and_retry

from cbc.lm.base import LMEngine
from cbc.utils.python import singleton


class Chatbot:
    """
    A class to interact with Google Bard.
    Parameters
        session_id: str
            The __Secure-1PSID cookie.
        proxy: str
    """

    __slots__ = [
        "headers",
        "_reqid",
        "SNlM0e",
        "conversation_id",
        "response_id",
        "choice_id",
        "session",
    ]

    def __init__(self, session_id: str, proxy: str = None):
        headers = {
            "Host": "bard.google.com",
            "X-Same-Domain": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Origin": "https://bard.google.com",
            "Referer": "https://bard.google.com/",
        }
        self._reqid = int("".join(random.choices(string.digits, k=4)))
        self.conversation_id = ""
        self.response_id = ""
        self.choice_id = ""
        self.session = requests.Session()
        if proxy:
            self.session.proxies.update(
                {
                    "http": proxy,
                    "https": proxy,
                },
            )
        self.session.headers = headers
        self.session.cookies.set("__Secure-1PSID", session_id)
        self.SNlM0e = self.__get_snlm0e()

    def __get_snlm0e(self):
        resp = self.session.get(url="https://bard.google.com/", timeout=10)
        # Find "SNlM0e":"<ID>"
        if resp.status_code != 200:
            raise Exception("Could not get Google Bard")
        return re.search(r"SNlM0e\":\"(.*?)\"", resp.text)[1]

    @sleep_and_retry
    @limits(calls=1, period=180)
    def ask(self, message: str) -> dict:
        """
        Send a message to Google Bard and return the response.
        :param message: The message to send to Google Bard.
        :return: A dict containing the response from Google Bard.
        """
        # url params
        params = {
            "bl": "boq_assistant-bard-web-server_20230514.20_p0",
            "_reqid": str(self._reqid),
            "rt": "c",
        }

        # message arr -> data["f.req"]. Message is double json stringified
        message_struct = [
            [message],
            None,
            [self.conversation_id, self.response_id, self.choice_id],
        ]
        data = {
            "f.req": json.dumps([None, json.dumps(message_struct)]),
            "at": self.SNlM0e,
        }

        # do the request!
        resp = self.session.post(
            "https://bard.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate",
            params=params,
            data=data,
            timeout=120,
        )

        chat_data = json.loads(resp.content.splitlines()[3])[0][2]
        if not chat_data:
            return {"content": f"Google Bard encountered an error: {resp.content}."}
        json_chat_data = json.loads(chat_data)
        results = {
            "content": json_chat_data[0][0],
            "conversation_id": json_chat_data[1][0],
            "response_id": json_chat_data[1][1],
            "factualityQueries": json_chat_data[3],
            "textQuery": json_chat_data[2][0] if json_chat_data[2] is not None else "",
            "choices": [{"id": i[0], "content": i[1]} for i in json_chat_data[4]],
        }
        self.conversation_id = results["conversation_id"]
        self.response_id = results["response_id"]
        self.choice_id = results["choices"][0]["id"]
        self._reqid += 100000
        return results


@singleton
class BardEngine(LMEngine):
    DEFAULT_POSTPROCESSOR = "all_truncate"

    def __init__(
        self,
    ) -> None:
        self._chatbot = Chatbot(session_id=os.environ.get("GOOGLE_BARD_SESSION_ID", None))

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:
        # Filter the prompt (one-off experiment)
        new_prompt = prompt.replace(
            "Summary:",
            "\nComplete the following sentence according to the task above. Output the completion as JSON with the key 'detailed_completion'.\n",
        )
        new_prompt = (
            new_prompt.replace("I'm not sure, but the image is likely of", "").strip()
            + "\nSentence: I'm not sure, but the captions likely describe"
        )
        # REPLACE ANYTHING EVEN REMOTELY RELATED TO IMAGES/PICTURES WITH THE WORD "CAPTION"
        new_prompt = re.sub(
            r"image|picture|photo|photograph|drawing|painting|illustration", "caption", new_prompt, flags=re.IGNORECASE
        )

        outputs = []
        for _ in range(n_completions):
            for i in range(5):
                logging.info("Sending prompt to BARD engine: %s", new_prompt)
                bard_result = self._chatbot.ask(new_prompt)["content"]
                try:
                    # Get the completion from any JSON embedded in the output. First, get content between the first
                    # pair of curly braces. Then, parse the JSON and get the value of the key "completion".
                    logging.info(f"Got output from BARD engine: {bard_result}")
                    output = bard_result[bard_result.find("{") : bard_result.find("}") + 1]
                    output = json.loads(output)["detailed_completion"]
                    outputs.append(output)
                    break
                except (json.JSONDecodeError, IndexError):
                    logging.error(f"Error parsing output from BARD engine: {bard_result}... retrying.")
                    time.sleep(i**1.5)
                    continue
            else:
                output = self._chatbot.ask(prompt)["content"]
                outputs.append(output)

        if len(outputs) < n_completions:
            # Pad with empty strings
            logging.warning("BardEngine returned fewer completions than requested.")
            outputs.extend([""] * (n_completions - len(outputs)))

        return outputs

    def best(self, prompt: str) -> str:
        return self(prompt, n_completions=1, temperature=0.0)[0]
