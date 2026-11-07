"""Hand-weighted transformer circuits for exact decimal arithmetic.

    from circuits import arith
    arith.evaluate("12.5", "*", "-0.04")   # '-0.500'
    arith.div("1", "7", places=10)         # '0.1428571428'
"""
