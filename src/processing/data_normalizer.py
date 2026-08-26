import re
from datetime import datetime
from typing import Dict, Optional


class DataNormalizer:
    """
    Normalize raw entities extracted from LayoutLMv3.

    Input:
        {
            "company": "...",
            "date": "...",
            "address": "...",
            "total": "..."
        }

    Output:
        {
            "company": "...",
            "date": "YYYY-MM-DD",
            "time": "HH:MM",
            "address": "...",
            "total": float,
            "currency": "MYR"
        }
    """

    def normalize(
        self,
        entities: Dict[str, str]
    ) -> Dict:

        normalized = {
            "company": self.normalize_text(
                entities.get("company", "")
            ),

            "date": self.normalize_date(
                entities.get("date", "")
            ),

            "time": self.normalize_time(
                entities.get("date", "")
            ),

            "address": self.normalize_text(
                entities.get("address", "")
            ),

            "total": self.normalize_amount(
                entities.get("total", "")
            ),

            "currency": "MYR"
        }

        return normalized
    @staticmethod
    def normalize_text(
        text: str
    ) -> str:

        if not text:
            return ""

        # Remove duplicated spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    @staticmethod
    def normalize_date(
        text: str
    ) -> Optional[str]:

        if not text:
            return None

        # Find date:
        # 22/12/2017
        # 22-12-2017
        # 22.12.2017

        match = re.search(
            r"(\d{1,2})[\/\-.]"
            r"(\d{1,2})[\/\-.]"
            r"(\d{4})",
            text
        )

        if not match:
            return None

        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))

        try:

            date = datetime(year, month, day)
            return date.strftime("%Y-%m-%d")

        except ValueError:
            return None

    @staticmethod
    def normalize_time(
        text: str
    ) -> Optional[str]:

        if not text:
            return None

        match = re.search(
            r"(\d{1,2}):(\d{2})",
            text
        )

        if not match:
            return None

        hour = int(match.group(1))
        minute = int(match.group(2))

        if (
            hour < 0
            or hour > 23
            or minute < 0
            or minute > 59
        ):
            return None

        return f"{hour:02d}:{minute:02d}"


    @staticmethod
    def normalize_amount(
        text: str
    ) -> Optional[float]:

        if not text:
            return None

        # Remove currency symbols / letters
        cleaned = re.sub(
            r"[^\d.,\-]",
            "",
            text
        )

        if not cleaned:
            return None
        
        # Handle common formats:
        # 15.90
        # 1,234.56
        # 1.234,56
        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                # European format
                # 1.234,56
                cleaned = cleaned.replace(
                    ".",
                    ""
                )
                cleaned = cleaned.replace(
                    ",",
                    "."
                )
            else:
                # Standard format
                # 1,234.56
                cleaned = cleaned.replace(
                    ",",
                    ""
                )

        elif "," in cleaned:
            # Assume comma is decimal separator
            cleaned = cleaned.replace(
                ",",
                "."
            )

        try:
            return float(cleaned)

        except ValueError:
            return None