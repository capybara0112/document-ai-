import re
from decimal import Decimal, InvalidOperation


def normalize_vietnamese_amount(value: str) -> Decimal | None:
    """Parse common Vietnamese amount formats; ambiguous formats return None."""
    if not value:
        return None

    cleaned = re.sub(r"(?i)\s*(vnd|vnđ|đ|₫)\s*", "", value.strip())
    cleaned = cleaned.replace(" ", "")
    if not re.fullmatch(r"[-+]?\d[\d.,]*", cleaned):
        return None

    # Common invoice format: 1.500.000 or 1,500,000.
    if re.fullmatch(r"[-+]?\d{1,3}(?:[.,]\d{3})+", cleaned):
        cleaned = cleaned.replace(".", "").replace(",", "")
    elif "." in cleaned and "," in cleaned:
        # Treat the last separator as decimal separator.
        decimal_separator = "." if cleaned.rfind(".") > cleaned.rfind(",") else ","
        thousands_separator = "," if decimal_separator == "." else "."
        cleaned = cleaned.replace(thousands_separator, "")
        cleaned = cleaned.replace(decimal_separator, ".")
    elif "," in cleaned:
        # A single comma followed by 1-2 digits is treated as decimal.
        tail = cleaned.rsplit(",", 1)[-1]
        cleaned = cleaned.replace(",", ".") if len(tail) in (1, 2) else cleaned.replace(",", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None
