import difflib
from datetime import datetime

def _payee_similarity(receipt_payee: str, candidate_payee: str, candidate_imported_payee: str) -> float:
    """
    Fuzzy-match the receipt payee against both the user-facing payee name and the
    raw bank-provided imported_payee.  Returns the best score (0.0–1.0).
    """
    receipt_lower = receipt_payee.lower().strip()
    scores = []
    for name in [candidate_payee, candidate_imported_payee]:
        if not name:
            continue
        name_lower = name.lower().strip()
        # Full SequenceMatcher ratio
        ratio = difflib.SequenceMatcher(None, receipt_lower, name_lower).ratio()
        scores.append(ratio)
        # Also check if one contains the other (common for "WALMART" vs "WALMART SUPERCENTER #1234")
        if receipt_lower in name_lower or name_lower in receipt_lower:
            scores.append(max(ratio, 0.80))
    return max(scores) if scores else 0.0


def _amount_similarity(receipt_cents: int, candidate_cents: int) -> float:
    """
    Score how close the two amounts are.  Both values should be absolute
    (positive) cent values.  Returns 0.0–1.0.
    """
    if receipt_cents == 0 and candidate_cents == 0:
        return 1.0
    if receipt_cents == 0 or candidate_cents == 0:
        return 0.0
    ratio = min(receipt_cents, candidate_cents) / max(receipt_cents, candidate_cents)
    # Exact match bonus
    if receipt_cents == candidate_cents:
        return 1.0
    return ratio


def _date_proximity(receipt_date: str, candidate_date: str, max_days: int = 7) -> float:
    """
    Linear decay over *max_days*.  Same-day → 1.0, max_days away → 0.0.
    """
    try:
        rd = datetime.strptime(receipt_date, "%Y-%m-%d")
        cd = datetime.strptime(candidate_date, "%Y-%m-%d")
        delta = abs((rd - cd).days)
        if delta == 0:
            return 1.0
        if delta >= max_days:
            return 0.0
        return 1.0 - (delta / max_days)
    except (ValueError, TypeError):
        return 0.0


def score_candidate(receipt_payee: str, receipt_cents: int, receipt_date: str, candidate: dict) -> float:
    """
    Return an aggregate match score (0.0–1.0) for a single Actual Budget
    transaction candidate against the parsed receipt.
    
    Weights:
      - Amount similarity:  0.60
      - Date proximity:     0.30
      - Payee similarity:   0.10
    """
    W_AMOUNT = 0.60
    W_DATE   = 0.30
    W_PAYEE  = 0.10

    cand_amount = abs(candidate.get("amount", 0))
    amount_score = _amount_similarity(receipt_cents, cand_amount)

    payee_score = _payee_similarity(
        receipt_payee,
        candidate.get("payee.name") or candidate.get("payee_name", ""),
        candidate.get("imported_payee", ""),
    )

    cand_date = candidate.get("date", "")
    date_score = _date_proximity(receipt_date, cand_date)

    return (W_AMOUNT * amount_score) + (W_PAYEE * payee_score) + (W_DATE * date_score)
