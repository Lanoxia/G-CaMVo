"""Small numerical routines to keep the core free from a SciPy dependency."""

from __future__ import annotations

import math


def sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def logit(probability: float) -> float:
    if not 0 < probability < 1:
        raise ValueError("probability must be in (0, 1)")
    return math.log(probability) - math.log1p(-probability)


def beta_log_pdf(value: float, alpha: float, beta: float) -> float:
    if not 0 < value < 1:
        raise ValueError("Beta density value must be in (0, 1)")
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta shape parameters must be positive")
    log_normalizer = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (
        (alpha - 1.0) * math.log(value)
        + (beta - 1.0) * math.log1p(-value)
        - log_normalizer
    )


def _beta_continued_fraction(alpha: float, beta: float, value: float) -> float:
    """Lentz evaluation of the incomplete-Beta continued fraction."""

    max_iterations = 300
    tolerance = 3e-14
    minimum = 1e-300
    qab = alpha + beta
    qap = alpha + 1.0
    qam = alpha - 1.0
    c = 1.0
    d = 1.0 - qab * value / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration
            * (beta - iteration)
            * value
            / ((qam + doubled) * (alpha + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (alpha + iteration)
            * (qab + iteration)
            * value
            / ((alpha + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= tolerance:
            return result
    raise FloatingPointError("incomplete-Beta continued fraction did not converge")


def regularized_beta_cdf(value: float, alpha: float, beta: float) -> float:
    """Regularized incomplete beta I_x(alpha, beta), i.e. the Beta CDF."""

    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta shape parameters must be positive")
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    log_front = (
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + alpha * math.log(value)
        + beta * math.log1p(-value)
    )
    front = math.exp(log_front)
    if value < (alpha + 1.0) / (alpha + beta + 2.0):
        result = front * _beta_continued_fraction(alpha, beta, value) / alpha
    else:
        result = 1.0 - (
            front * _beta_continued_fraction(beta, alpha, 1.0 - value) / beta
        )
    return min(1.0, max(0.0, result))

