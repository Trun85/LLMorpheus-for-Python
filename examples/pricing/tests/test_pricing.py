"""A deliberately incomplete test suite.

It covers the happy paths but leaves several boundaries untested, so mutation
testing has something to find.
"""

from sample_pkg import pricing

BASKET = [
    {"name": "mug", "price": 12.0, "quantity": 2},
    {"name": "poster", "price": 8.5, "quantity": 1},
]


def test_subtotal():
    assert pricing.subtotal(BASKET) == 32.5


def test_subtotal_defaults_quantity_to_one():
    assert pricing.subtotal([{"name": "pen", "price": 3.0}]) == 3.0


def test_gold_tier_gets_fifteen_percent():
    assert pricing.discount_rate("gold", 10) == 0.15


def test_standard_tier_gets_nothing_on_small_orders():
    assert pricing.discount_rate("standard", 10) == 0.0


def test_shipping_is_free_over_the_threshold():
    assert pricing.shipping_cost(80.0) == 0.0


def test_shipping_is_charged_under_the_threshold():
    assert pricing.shipping_cost(10.0) == 4.95


def test_total_adds_shipping():
    assert pricing.total(BASKET) == 37.45


def test_cheapest_returns_none_for_empty_basket():
    assert pricing.cheapest([]) is None


def test_cheapest_picks_lowest_price():
    assert pricing.cheapest(BASKET)["name"] == "poster"


def test_names_over_filters():
    assert pricing.names_over(BASKET, 10.0) == ["mug"]
