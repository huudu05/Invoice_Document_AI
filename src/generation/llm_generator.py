import os

from openai import OpenAI


class LLMGenerator:
    """
    Generate answers using Gemini API
    through OpenAI-compatible interface.
    """

    def __init__(
        self,
        model: str = "gemini-3.6-flash",
    ):

        self.model = model

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable "
                "is not set."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            ),
        )

    def generate(
        self,
        prompt: str,
    ) -> str:

        if not prompt or not prompt.strip():
            return ""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = (
            response.choices[0]
            .message
            .content
        )

        if not answer:
            return ""

        return answer.strip()