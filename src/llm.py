import yaml
import time

from anthropic import Anthropic
from together import Together
from openai import OpenAI
from google import genai
from google.genai import types


class LLM:
    def __init__(self,
                 model="claude",
                 config_path="configs/secrets.yml",
                 temperature=0,
                 max_tokens=1000,
                 retry_time=1000,
                 failed_sleep_time=1,
                 round_sleep_time=1):

        self.model = model
        self.config_path = config_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_time = retry_time
        self.failed_sleep_time = failed_sleep_time
        self.round_sleep_time = round_sleep_time
        self.secrets = None
        self.client = None

        with open(self.config_path) as f:
            self.secrets = yaml.safe_load(f)

        if self.model == "claude":
            self.client = Anthropic(api_key=self.secrets.get("claude"))
        elif self.model == "llama" or self.model == "mixtral":
            self.client = Together(api_key=self.secrets.get("together"))
        elif self.model == "gpt-4o":
            self.client = OpenAI(api_key=self.secrets.get("openai"))
        elif self.model == "gemini":
            self.client = genai.Client(api_key=self.secrets.get("google"))

    def generate(self, prompt):
        message = None
        if self.model == "claude":
            message = self.claude_completion(prompt=prompt)
        elif self.model == "llama":
            message = self.llama_completion(prompt=prompt)
        elif self.model == "mixtral":
            message = self.mixtral_completion(prompt=prompt)
        elif self.model == "gpt-4o":
            message = self.gpt_4o_completion(prompt=prompt)
        elif self.model == "gemini":
            message = self.gemini_completion(prompt=prompt)

        return message

    def gemini_completion(self, prompt):
        message = None
        try:
            message = self.client.models.generate_content(
                model="gemini-1.5-pro",
                contents=prompt.get("user"),
                config=types.GenerateContentConfig(
                    system_instruction=prompt.get("system"),
                    temperature=self.temperature,
                    response_mime_type='application/json',
                    stop_sequences=['\n'],
                    safety_settings=[
                        types.SafetySetting(
                            category='HARM_CATEGORY_HARASSMENT',
                            threshold='OFF'
                        ),
                        types.SafetySetting(
                            category='HARM_CATEGORY_HATE_SPEECH',
                            threshold='OFF'
                        ),
                        types.SafetySetting(
                            category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                            threshold='OFF'
                        ),
                        types.SafetySetting(
                            category='HARM_CATEGORY_DANGEROUS_CONTENT',
                            threshold='OFF'
                        ),
                        types.SafetySetting(
                            category='HARM_CATEGORY_CIVIC_INTEGRITY',
                            threshold='OFF'
                        ),
                    ]
                ),
            )
        except Exception as e:
            print(e)
            for retry_time in range(self.retry_time):
                retry_time = retry_time + 1
                print(f"{self.model} Retry {retry_time}")
                time.sleep(self.failed_sleep_time)
                try:
                    message = self.client.models.generate_content(
                        model="gemini-1.5-pro",
                        contents=prompt.get("user"),
                        config=types.GenerateContentConfig(
                            system_instruction=prompt.get("system"),
                            temperature=self.temperature,
                            response_mime_type='application/json',
                            stop_sequences=['\n'],
                            safety_settings=[
                                types.SafetySetting(
                                    category='HARM_CATEGORY_HARASSMENT',
                                    threshold='OFF'
                                ),
                                types.SafetySetting(
                                    category='HARM_CATEGORY_HATE_SPEECH',
                                    threshold='OFF'
                                ),
                                types.SafetySetting(
                                    category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                                    threshold='OFF'
                                ),
                                types.SafetySetting(
                                    category='HARM_CATEGORY_DANGEROUS_CONTENT',
                                    threshold='OFF'
                                ),
                                types.SafetySetting(
                                    category='HARM_CATEGORY_CIVIC_INTEGRITY',
                                    threshold='OFF'
                                ),
                            ]
                        ),
                    )
                    break
                except Exception as _:
                    continue

        time.sleep(self.round_sleep_time)
        return message.text

    def gpt_4o_completion(self, prompt):
        message = None
        try:
            message = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": prompt.get("system")},
                    {"role": "user", "content": prompt.get("user")},
                ],
                response_format={"type": "text"},
                temperature=self.temperature,
                # max_completion_tokens=self.max_tokens
            )
        except Exception as e:
            print(e)
            for retry_time in range(self.retry_time):
                retry_time = retry_time + 1
                print(f"{self.model} Retry {retry_time}")
                time.sleep(self.failed_sleep_time)
                try:
                    message = self.client.chat.completions.create(
                        model="gpt-4o-2024-08-06",
                        messages=[
                            {"role": "system", "content": prompt.get("system")},
                            {"role": "user", "content": prompt.get("user")},
                        ],
                        response_format={"type": "text"},
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens
                    )
                    break
                except Exception as _:
                    continue

        time.sleep(self.round_sleep_time)
        return message.choices[0].message.content

    def mixtral_completion(self, prompt):
        message = None
        try:
            message = self.client.chat.completions.create(
                model="mistralai/Mixtral-8x22B-Instruct-v0.1",
                messages=[
                    {"role": "system", "content": prompt.get("system")},
                    {"role": "user", "content": prompt.get("user")},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["</s>","[/INST]"],
                stream=False
            )
        except Exception as e:
            print(e)
            for retry_time in range(self.retry_time):
                retry_time = retry_time + 1
                print(f"{self.model} Retry {retry_time}")
                time.sleep(self.failed_sleep_time)
                try:
                    message = self.client.chat.completions.create(
                        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
                        messages=[
                            {"role": "system", "content": prompt.get("system")},
                            {"role": "user", "content": prompt.get("user")},
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stop=["</s>","[/INST]"],
                        stream=False
                    )
                    break
                except Exception as _:
                    continue

        time.sleep(self.round_sleep_time)
        return message.choices[0].message.content

    def llama_completion(self, prompt):
        message = None
        try:
            message = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": prompt.get("system")},
                    {"role": "user", "content": prompt.get("user")},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["<|eot_id|>", "<|eom_id|>"],
                stream=False
            )
        except Exception as e:
            print(e)
            for retry_time in range(self.retry_time):
                retry_time = retry_time + 1
                print(f"{self.model} Retry {retry_time}")
                time.sleep(self.failed_sleep_time)
                try:
                    message = self.client.chat.completions.create(
                        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                        messages=[
                            {"role": "system", "content": prompt.get("system")},
                            {"role": "user", "content": prompt.get("user")},
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stop=["<|eot_id|>", "<|eom_id|>"],
                        stream=False
                    )
                    break
                except Exception as _:
                    continue

        time.sleep(self.round_sleep_time)
        return message.choices[0].message.content

    def claude_completion(self, prompt):
        message = None
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=prompt.get("system"),
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt.get("user")}]}]
            )
        except Exception as e:
            print(e)
            for retry_time in range(self.retry_time):
                retry_time = retry_time + 1
                print(f"{self.model} Retry {retry_time}")
                time.sleep(self.failed_sleep_time)
                try:
                    message = self.client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=prompt.get("system"),
                        messages=[{"role": "user", "content": [{"type": "text", "text": prompt.get("user")}]}]
                    )
                    break
                except Exception as _:
                    continue

        time.sleep(self.round_sleep_time)
        return message.content[0].text
