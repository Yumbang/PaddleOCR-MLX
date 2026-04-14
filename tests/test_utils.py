"""Tests for utility functions."""


from mlx_ppocr.models.backbone.pplcnetv3 import make_divisible


class TestMakeDivisible:
    def test_already_divisible(self):
        assert make_divisible(64, 8) == 64

    def test_rounds_up(self):
        # 65 → rounds to 64 or 72 depending on rounding
        result = make_divisible(65, 8)
        assert result % 8 == 0
        assert result >= 65 * 0.9

    def test_min_value_enforced(self):
        assert make_divisible(1, 8) == 8

    def test_custom_min_value(self):
        assert make_divisible(1, 8, min_value=16) == 16

    def test_default_min_is_divisor(self):
        assert make_divisible(0.5, 8) == 8

    def test_large_value(self):
        result = make_divisible(1000, 8)
        assert result % 8 == 0
        assert abs(result - 1000) <= 8

    def test_safety_margin(self):
        # If new_value < 0.9 * value, add divisor
        # value=17, divisor=8: int(17+4)//8*8 = 16, 16 < 15.3? No. → 16
        assert make_divisible(17, 8) == 16
        # value=9, divisor=8: int(9+4)//8*8 = 8, 8 < 8.1 → Yes → 16
        assert make_divisible(9, 8) == 16

    def test_divisor_of_1(self):
        assert make_divisible(42, 1) == 42
