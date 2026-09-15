import re
from datetime import datetime
from typing import Dict, Any
 
 
class InvoiceNormalizer:
    """
    Normalize invoice data after LayoutLMv3 has extracted it
    """
 
    MONTH_ABBREVIATIONS = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
 
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
 
        if not value:
            return ""
 
        original = value.strip()
 
        def try_parse(day: int, month: int, year: int):
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                return None

        match = re.search(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", original)
        if match:
            day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if 2000 <= year <= 2099:
                result = try_parse(day, month, year)
                if result:
                    return result
 
        match = re.search(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})", original)
        if match:
            day, month = int(match.group(1)), int(match.group(2))
            year = int(match.group(3)) + 2000
            result = try_parse(day, month, year)
            if result:
                return result
 
        match = re.search(
            r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{2,4})",
            original,
        )
 
        if match:
            day = int(match.group(1))
            month_name = match.group(2).lower()[:3]
            year_raw = match.group(3)
            month = InvoiceNormalizer.MONTH_ABBREVIATIONS.get(month_name)
 
            if month:
                year = int(year_raw)
                if year < 100:
                    year += 2000
 
                result = try_parse(day, month, year)
                if result:
                    return result
 
        return original
 
    @staticmethod
    def normalize_amount(value: str) -> float | None:
 
        if not value:
            return None
 
        value = str(value).strip()
 
        tokens = value.split()
        if len(tokens) == 2:
            first_clean = re.sub(r"[^\d.,-]", "", tokens[0])
            second_clean = re.sub(r"[^\d.,-]", "", tokens[1])
 
            first_is_full_amount = bool(re.fullmatch(r"\d+[.,]\d{1,2}", first_clean))
            second_is_full_amount = bool(re.fullmatch(r"\d+[.,]\d{1,2}", second_clean))
            first_is_plain_int = bool(re.fullmatch(r"\d+", first_clean))
            second_looks_like_cents = bool(re.fullmatch(r"\d{1,2}", second_clean))
 
            if first_is_full_amount and second_is_full_amount:
                value = tokens[1]
            elif first_is_plain_int and second_looks_like_cents:
                value = f"{first_clean}.{second_clean}"
 
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