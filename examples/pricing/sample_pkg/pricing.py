"""A small module with a deliberately under-tested test suite."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

FREE_SHIPPING_THRESHOLD = 50.0
SHIPPING_COST = 4.95


def subtotal(items: Iterable[Dict[str, float]]) -> float:
    """Sum ``price * quantity`` over the basket."""
    total = 0.0
    for item in items:
        total += item["price"] * item.get("quantity", 1)
    return round(total, 2)


def discount_rate(customer_tier: str, order_total: float) -> float:
    """Return the fraction of the order total to knock off."""
    if customer_tier == "gold":
        return 0.15
    if customer_tier == "silver" and order_total >= 100:
        return 0.10
    if order_total >= 200:
        return 0.05
    return 0.0


def shipping_cost(order_total: float, country: str = "US") -> float:
    if order_total >= FREE_SHIPPING_THRESHOLD:
        return 0.0
    if country != "US":
        return SHIPPING_COST * 2
    return SHIPPING_COST


def total(
    items: Iterable[Dict[str, float]],
    customer_tier: str = "standard",
    country: str = "US",
) -> float:
    """Basket total after discount and shipping."""
    items = list(items)
    base = subtotal(items)
    discounted = base * (1 - discount_rate(customer_tier, base))
    return round(discounted + shipping_cost(discounted, country), 2)


def cheapest(items: Iterable[Dict[str, float]]) -> Optional[Dict[str, float]]:
    items = list(items)
    if not items:
        return None
    return min(items, key=lambda item: item["price"])


def names_over(items: Iterable[Dict[str, float]], threshold: float) -> List[str]:
    return [str(item["name"]) for item in items if item["price"] > threshold]
