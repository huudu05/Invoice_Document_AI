import re
from datetime import datetime
from typing import Dict, Any


class InvoiceNormalizer:
    """
    Normalize invoice data after LayoutLMv3 has extracted it
    """

    @staticmethod
    def normalize_company(value: str) -> str:
        if not value:
            return ""

        value = re.sub(r"\s+", " ", value)
        return value.strip()

    @staticmethod
    def normalize_address(value: str) -> str:
        if not value:
            return ""

        value = re.sub(r"\s+", " ", value)
        return value.strip()

    @staticmethod
    def normalize_date(value: str) -> str:
        """
        Chuyển các dạng ngày phổ biến về YYYY-MM-DD.

        Ví dụ:
        22/12/2017
        22/12/201714:03
        -> 2017-12-22
        """

        if not value:
            return ""

        #dd/mm/yyyy
        match = re.search(
            r"(\d{1,2})/(\d{1,2})/(\d{4})",
            value
        )

        if match:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))

            try:
                date_obj = datetime(year, month, day)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                pass

        #dd-mm-yyyy
        match = re.search(
            r"(\d{1,2})-(\d{1,2})-(\d{4})",
            value
        )

        if match:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))

            try:
                date_obj = datetime(year, month, day)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                pass

        return value.strip()

    @staticmethod
    def normalize_amount(value: str) -> float | None:
        """
        "15.90" -> 15.90
        "RM 15.90" -> 15.90
        "1,250.50" -> 1250.50
        """

        if not value:
            return None

        value = str(value)

        cleaned = re.sub(r"[^\d.,-]", "", value)

        if not cleaned:
            return None

        #1,250.50
        if "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(",", "")

        #15,90
        elif "," in cleaned and "." not in cleaned:
            parts = cleaned.split(",")

            if len(parts) == 2 and len(parts[1]) <= 2:
                cleaned = ".".join(parts)
            else:
                cleaned = cleaned.replace(",", "")

        try:
            return float(cleaned)
        except ValueError:
            return None

    def normalize(self, structured: Dict[str, Any]) -> Dict[str, Any]:

        normalized = {
            "company": self.normalize_company(
                structured.get("company", "")
            ),

            "date": self.normalize_date(
                structured.get("date", "")
            ),

            "address": self.normalize_address(
                structured.get("address", "")
            ),

            "total": self.normalize_amount(
                structured.get("total", "")
            )
        }

        return normalized