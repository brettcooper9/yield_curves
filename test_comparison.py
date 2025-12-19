"""
Simple Python test script to compare with R outputs.
Run this with: python test_comparison.py
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from yield_curves.helpers import monthly_bond_return, round_month_to_maturity

print("=" * 60)
print("Python Test Outputs for R Comparison")
print("=" * 60)

# Test 1: Monthly Bond Return
print("\nTest 1: Monthly Bond Return")
print("-" * 30)
test_yields = np.array([2.5, 2.6, 2.4, 2.7, 2.5])
print(f"Yields: {test_yields}")
print()

for i in range(1, len(test_yields)):
    ret = monthly_bond_return(test_yields, i)
    print(f"Month {i}: {ret:.10f} ({ret*100:.6f}%)")

# Test 2: Time to Maturity
print("\n\nTest 2: Time to Maturity")
print("-" * 30)
test_cases = [
    ('2020-01-15', '2025-01-15'),
    ('2020-01-15', '2020-07-15'),
    ('2018-12-31', '2026-12-31'),
]

for current, maturity in test_cases:
    ttm = round_month_to_maturity(current, maturity, num_digits=4)
    print(f"{current} to {maturity}: {ttm:.4f} years")

# Test 3: Comprehensive Bond Return Test
print("\n\nTest 3: Comprehensive Bond Return Test")
print("-" * 30)
comp_yields = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
print(f"Yields: {comp_yields}")
print()

print(f"{'Index':<8} {'Prior_Yield':<12} {'Current_Yield':<14} {'Return':<20}")
print("-" * 60)
for i in range(1, len(comp_yields)):
    ret = monthly_bond_return(comp_yields, i)
    print(f"{i:<8} {comp_yields[i-1]:<12.1f} {comp_yields[i]:<14.1f} {ret:<20.10f}")

print("\n" + "=" * 60)
print("Python test outputs complete!")
print("Compare these results with your R console outputs.")
print("=" * 60)
