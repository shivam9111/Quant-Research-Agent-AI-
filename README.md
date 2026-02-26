# Quant Research Agent

A free, open-source quantitative finance research assistant that combines deterministic pricing engines with a lightweight open-source conversational model for interactive financial modelling and analysis.

---

## Overview

Quant Research Agent is a modular research tool designed to bridge classical quantitative finance models with conversational AI.

It integrates:

- Deterministic pricing models (Black-Scholes)
- Implied volatility extraction
- Numerical methods
- Conversational reasoning via open-source LLMs
- Interactive research workflow

The system is designed as a personal quant research lab rather than a production trading system.

---

## Features

### Quantitative Engine
- Black-Scholes option pricing (call/put)
- Greeks computation
- Implied volatility (Newton-Raphson)
- Deterministic numerical modelling

### Conversational Interface
- Natural language interaction
- Context memory across conversation
- Model explanations and theory discussion
- Integrated routing between computation and reasoning

### Architecture
- Modular design
- Separation of computation and language layers
- Easily extensible for additional models

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/quant-research-agent.git
cd quant-research-agent
