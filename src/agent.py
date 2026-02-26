# =============================================================================
# Quant Research AI Agent
# =============================================================================
# An AI-powered research assistant combining a quantitative finance engine
# (Black-Scholes pricing, Greeks, implied volatility) with a local LLM
# (TinyLlama) for natural language financial research conversations.
#
# The agent uses intent detection to route user queries: quantitative inputs
# are handled analytically; open-ended research questions are passed to the
# language model.
#
# Author: Shivam Gujral
# =============================================================================

import re
import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import brentq
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# =============================================================================
# 1. QUANTITATIVE FINANCE ENGINE
# =============================================================================

class QuantEngine:
    """
    Analytical pricing and risk engine for vanilla options.

    Methods
    -------
    black_scholes     : Price a European call or put and compute Greeks.
    implied_volatility: Recover implied volatility from a market price via
                        Newton-Raphson iteration with Brent's method fallback.
    binomial_price    : Price a European or American option using a
                        Cox-Ross-Rubinstein binomial tree.
    """

    def black_scholes(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call"
    ) -> dict:
        """
        Price a European option and compute first-order Greeks using
        the Black-Scholes-Merton formula.

        Parameters
        ----------
        S           : Current underlying price.
        K           : Strike price.
        T           : Time to expiry in years.
        r           : Continuously compounded risk-free rate.
        sigma       : Annualised volatility.
        option_type : 'call' or 'put'.

        Returns
        -------
        dict with keys: price, delta, gamma, vega, theta, rho.
        """
        if T <= 0:
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            return {"price": intrinsic, "delta": None, "gamma": None,
                    "vega": None, "theta": None, "rho": None}

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega  = S * norm.pdf(d1) * np.sqrt(T) / 100

        return {
            "price": round(price, 4),
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "vega":  round(vega,  4),
            "theta": round(theta, 4),
            "rho":   round(rho,   4)
        }

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        initial_guess: float = 0.2
    ) -> float:
        """
        Recover implied volatility from a market price using Newton-Raphson
        with Brent's method as a fallback for robustness.

        Parameters
        ----------
        market_price  : Observed market price of the option.
        S, K, T, r    : Standard Black-Scholes inputs.
        option_type   : 'call' or 'put'.
        initial_guess : Starting point for Newton-Raphson (default 0.2).

        Returns
        -------
        float : Implied volatility.
        """
        def objective(sigma):
            return self.black_scholes(S, K, T, r, sigma, option_type)["price"] - market_price

        # Newton-Raphson
        sigma = initial_guess
        for _ in range(100):
            bs = self.black_scholes(S, K, T, r, sigma, option_type)
            vega = bs["vega"] * 100  # undo /100 scaling
            if abs(vega) < 1e-10:
                break
            sigma -= (bs["price"] - market_price) / vega
            if sigma <= 0:
                sigma = 1e-6

        # Fallback to Brent's method if Newton-Raphson diverges
        if sigma <= 0 or sigma > 10:
            try:
                sigma = brentq(objective, 1e-6, 10.0)
            except ValueError:
                return float("nan")

        return round(sigma, 6)

    def binomial_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n: int = 100,
        option_type: str = "call",
        style: str = "european"
    ) -> float:
        """
        Price a European or American option using a Cox-Ross-Rubinstein
        binomial tree.

        Parameters
        ----------
        S, K, T, r, sigma : Standard Black-Scholes inputs.
        n                  : Number of time steps (default 100).
        option_type        : 'call' or 'put'.
        style              : 'european' or 'american'.

        Returns
        -------
        float : Option price.
        """
        dt = T / n
        u  = np.exp(sigma * np.sqrt(dt))
        d  = 1 / u
        p  = (np.exp(r * dt) - d) / (u - d)

        # Terminal asset prices
        ST = S * u ** np.arange(n, -1, -1) * d ** np.arange(0, n + 1)

        # Terminal payoffs
        if option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        # Backward induction
        discount = np.exp(-r * dt)
        for i in range(n - 1, -1, -1):
            payoffs = discount * (p * payoffs[:-1] + (1 - p) * payoffs[1:])
            if style == "american":
                ST = S * u ** np.arange(i, -1, -1) * d ** np.arange(0, i + 1)
                intrinsic = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)
                payoffs = np.maximum(payoffs, intrinsic)

        return round(payoffs[0], 4)


# =============================================================================
# 2. INTENT DETECTION
# =============================================================================

def detect_intent(text: str) -> str:
    """
    Classify user input into one of three intents:
    - 'black_scholes'       : Request to price an option or compute Greeks.
    - 'implied_volatility'  : Request to compute implied volatility.
    - 'binomial'            : Request to use a binomial tree.
    - 'chat'                : General research question for the LLM.

    Parameters
    ----------
    text : Raw user input (case-insensitive).

    Returns
    -------
    str : One of the intent labels above.
    """
    t = text.lower()
    if "implied" in t and ("vol" in t or "volatility" in t):
        return "implied_volatility"
    if "binomial" in t or "american" in t or "crr" in t:
        return "binomial"
    if ("black" in t and "scholes" in t) or "price" in t or "greeks" in t or "delta" in t:
        return "black_scholes"
    return "chat"


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a string."""
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


# =============================================================================
# 3. RESEARCH ASSISTANT AGENT
# =============================================================================

class ResearchAssistant:
    """
    Conversational research assistant combining quantitative computation
    with LLM-backed natural language responses.

    Quantitative queries (Black-Scholes, implied vol, binomial pricing)
    are routed to the QuantEngine. General research questions are handled
    by the language model with full conversation history.

    Parameters
    ----------
    engine        : QuantEngine instance.
    chat_pipeline : HuggingFace text-generation pipeline.
    """

    def __init__(self, engine: QuantEngine, chat_pipeline):
        self.engine  = engine
        self.chat    = chat_pipeline
        self.history = []  # list of (role, text) tuples

    def _format_conversation(self, prompt: str) -> str:
        """Build prompt string with full conversation history."""
        conversation = ""
        for role, text in self.history:
            conversation += f"{role}: {text}\n"
        conversation += f"user: {prompt}\nassistant:"
        return conversation

    def _llm_response(self, prompt: str) -> str:
        """Generate a response from the LLM and update history."""
        conversation = self._format_conversation(prompt)
        output = self.chat(
            conversation,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        response = output[0]["generated_text"].split("assistant:")[-1].strip()
        self.history.append(("user", prompt))
        self.history.append(("assistant", response))
        return response

    def respond(self, user_input: str) -> str:
        """
        Route user input to the appropriate handler and return a response.

        Parameters
        ----------
        user_input : Raw user query.

        Returns
        -------
        str : Formatted response string.
        """
        intent  = detect_intent(user_input)
        numbers = extract_numbers(user_input)

        if intent == "black_scholes":
            if len(numbers) >= 5:
                S, K, T, r, sigma = numbers[:5]
                option_type = "put" if "put" in user_input.lower() else "call"
                result = self.engine.black_scholes(S, K, T, r, sigma, option_type)
                return self._format_bs_result(result, option_type, S, K, T, r, sigma)
            return (
                "Please provide 5 parameters: S (spot), K (strike), T (time to expiry), "
                "r (risk-free rate), sigma (volatility).\n"
                "Example: 'Black-Scholes 100 105 0.5 0.05 0.2'"
            )

        if intent == "implied_volatility":
            if len(numbers) >= 5:
                market_price, S, K, T, r = numbers[:5]
                option_type = "put" if "put" in user_input.lower() else "call"
                iv = self.engine.implied_volatility(market_price, S, K, T, r, option_type)
                return f"\nImplied Volatility: {iv:.4f} ({iv*100:.2f}%)"
            return (
                "Please provide 5 parameters: market_price, S, K, T, r.\n"
                "Example: 'Implied vol 10.5 100 105 0.5 0.05'"
            )

        if intent == "binomial":
            if len(numbers) >= 5:
                S, K, T, r, sigma = numbers[:5]
                option_type = "put" if "put" in user_input.lower() else "call"
                style = "american" if "american" in user_input.lower() else "european"
                price = self.engine.binomial_price(S, K, T, r, sigma, option_type=option_type, style=style)
                return f"\nBinomial Tree Price ({style.capitalize()} {option_type}): {price}"
            return (
                "Please provide 5 parameters: S, K, T, r, sigma.\n"
                "Example: 'Binomial American put 100 105 0.5 0.05 0.2'"
            )

        return self._llm_response(user_input)

    @staticmethod
    def _format_bs_result(result: dict, option_type: str, S, K, T, r, sigma) -> str:
        """Format Black-Scholes output as a readable string."""
        return (
            f"\n{'='*45}\n"
            f"  Black-Scholes Pricing Result\n"
            f"{'='*45}\n"
            f"  Inputs  : S={S}, K={K}, T={T}, r={r}, σ={sigma}\n"
            f"  Type    : {option_type.capitalize()}\n"
            f"{'─'*45}\n"
            f"  Price   : {result['price']}\n"
            f"  Delta   : {result['delta']}\n"
            f"  Gamma   : {result['gamma']}\n"
            f"  Vega    : {result['vega']}  (per 1% move in vol)\n"
            f"  Theta   : {result['theta']}  (per calendar day)\n"
            f"  Rho     : {result['rho']}   (per 1% move in r)\n"
            f"{'='*45}\n"
        )


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    print("\nLoading language model (TinyLlama-1.1B)...")

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    engine    = QuantEngine()
    assistant = ResearchAssistant(engine, chat_pipeline)

    print("\n" + "="*50)
    print("  Quant Research AI Agent")
    print("="*50)
    print("  Commands:")
    print("  - Black-Scholes: 'bs 100 105 0.5 0.05 0.2'")
    print("  - Implied vol:   'implied vol 10.5 100 105 0.5 0.05'")
    print("  - Binomial:      'binomial american put 100 105 0.5 0.05 0.2'")
    print("  - Research chat: any open-ended question")
    print("  - Type 'exit' to quit")
    print("="*50 + "\n")

    while True:
        user_input = input("Research AI > ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Exiting. Goodbye.")
            break
        print(assistant.respond(user_input))


if __name__ == "__main__":
    main()
