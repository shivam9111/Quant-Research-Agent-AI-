# Quant Research AI Agent

A hybrid quantitative finance research assistant combining a deterministic pricing engine with a local open-source language model for interactive financial modelling and research dialogue.

---

## Overview

Quant Research AI Agent bridges classical financial mathematics and conversational AI.

The system integrates:
- Closed-form Black-Scholes pricing
- Greeks computation
- Implied volatility extraction (Newton-Raphson with Brent fallback)
- Cox-Ross-Rubinstein binomial tree pricing (European & American)
- Local LLM-powered research dialogue (TinyLlama)
- Intent-based routing between analytics and natural language reasoning

This project is designed as a modular quantitative research laboratory rather than a production trading system.

---

## Features

### Quantitative Engine
- Black-Scholes option pricing (call & put)
- Full first-order Greeks (Delta, Gamma, Vega, Theta, Rho)
- Robust implied volatility solver with numerical safeguards
- Binomial tree pricing (CRR) for European and American options
- Deterministic analytical outputs

### Conversational Interface
- Natural language research interaction
- Context memory across conversation
- Automatic intent detection (pricing vs research discussion)
- Integrated routing between numerical engine and LLM

---

## Architecture Design

The system follows a modular architecture:

1. Intent Detection Layer  
   - Regex-based classification of user input  
   - Routes queries to either quantitative engine or LLM  

2. Quantitative Engine  
   - Closed-form analytical pricing  
   - Numerical root-finding for implied volatility  
   - Tree-based dynamic programming (CRR)  

3. Language Model Layer  
   - TinyLlama (1.1B) via HuggingFace  
   - Local inference (no external API dependency)  
   - Conversation state memory  

4. Execution Interface  
   - CLI-based interactive research workflow  
   - Structured formatted output for pricing results  

---

## Example Usage

Black-Scholes pricing:
bs 100 105 0.5 0.05 0.2

Implied volatility:
implied vol 10.5 100 105 0.5 0.05

Binomial pricing:
binomial american put 100 105 0.5 0.05 0.2

Research dialogue:
Explain why Vega peaks at-the-money.

---

## How to Run

1. Clone the repository.
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python src/agent.py

---

## Key Contributions

- Built a hybrid quant + AI system integrating deterministic finance models with local LLM inference.
- Implemented robust numerical methods for implied volatility recovery.
- Designed modular intent-based routing between analytical and conversational components.
- Developed a structured research CLI suitable for quantitative experimentation and extension.
