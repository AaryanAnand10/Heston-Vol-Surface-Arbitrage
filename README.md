# Heston-Vol-Surface-Arbitrage

# Heston Model Volatility Surface Arbitrage

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-1.20+-orange.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.7+-green.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

&gt; A complete quantitative derivatives research pipeline implementing the Heston stochastic volatility model for option pricing, calibration, and volatility surface arbitrage signal generation.

---

## Table of Contents

1. [Overview](#overview)
   - [The Problem with Black-Scholes](#the-problem-with-black-scholes)
   - [The Heston Solution](#the-heston-solution)
   - [Project Capabilities](#project-capabilities)
2. [Mathematical Foundation](#mathematical-foundation)
   - [The Two Coupled SDEs](#the-two-coupled-sdes)
   - [Parameter-by-Parameter Intuition](#parameter-by-parameter-intuition)
   - [The Leverage Effect](#the-leverage-effect)
   - [The Characteristic Function](#the-characteristic-function)
   - [Carr-Madan FFT Pricing](#carr-madan-fft-pricing)
   - [Monte Carlo with Milstein Scheme](#monte-carlo-with-milstein-scheme)
   - [Implied Volatility Extraction](#implied-volatility-extraction)
   - [Model Calibration](#model-calibration)
   - [Arbitrage Signal Generation](#arbitrage-signal-generation)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Quick Start](#quick-start)
   - [Running the Full Notebook](#running-the-full-notebook)
6. [Block-by-Block Implementation](#block-by-block-implementation)
   - [Block 0: Imports and Setup](#block-0-imports-and-setup)
   - [Block 1: Model Parameters](#block-1-model-parameters)
   - [Block 2: Characteristic Function](#block-2-characteristic-function)
   - [Block 3: Carr-Madan FFT Pricing](#block-3-carr-madan-fft-pricing)
   - [Block 4: Monte Carlo Simulation](#block-4-monte-carlo-simulation)
   - [Block 5: Implied Volatility Extraction](#block-5-implied-volatility-extraction)
   - [Block 6: Visualization](#block-6-visualization)
   - [Block 7: Model Calibration](#block-7-model-calibration)
   - [Block 8: Arbitrage Signal Generation](#block-8-arbitrage-signal-generation)
7. [Key Mathematical Results](#key-mathematical-results)
8. [Complete Data Flow](#complete-data-flow)
9. [Interview Talking Points](#interview-talking-points)
10. [Extensions and Next Steps](#extensions-and-next-steps)
11. [References](#references)
12. [License](#license)

---

## Overview

### The Problem with Black-Scholes

The Black-Scholes-Merton model (1973) revolutionized derivatives pricing by providing a closed-form formula for European options. However, it rests on a critical assumption: **volatility is constant**.

In reality, financial markets exhibit three phenomena that Black-Scholes cannot explain:

| Phenomenon | What We Observe | Why Black-Scholes Fails |
|------------|----------------|------------------------|
| **Volatility Smile/Skew** | OTM puts have higher IV than ATM | BS predicts flat IV across strikes |
| **Volatility Clustering** | High vol periods follow high vol periods | BS assumes i.i.d. returns |
| **Leverage Effect** | Markets crash faster than they rally | BS assumes symmetric log-normal distribution |

These phenomena are not academic curiosities — they represent **systematic mispricings** that quantitative trading strategies exploit.

### The Heston Solution

The **Heston stochastic volatility model (1993)** addresses this by making volatility a random process that evolves according to its own stochastic differential equation. Specifically:

- **Variance follows a mean-reverting CIR process** — it oscillates around a long-term average
- **Spot and volatility shocks are correlated** — this generates the leverage effect
- **The model is affine** — the characteristic function has a closed form, enabling fast Fourier pricing

This project implements the complete pipeline: from mathematical model to trading signals.

### Project Capabilities

| Capability | Method | Industry Application |
|------------|--------|---------------------|
| Fast European option pricing | Carr-Madan FFT | Real-time quoting, risk management |
| Path-dependent option pricing | Monte Carlo (Milstein) | Exotics pricing, barrier options |
| Model validation | FFT vs Monte Carlo cross-check | Model risk control |
| Market fitting | L-BFGS-B calibration | Daily parameter estimation |
| Trading signal generation | Vol surface deviation analysis | Volatility arbitrage strategies |

