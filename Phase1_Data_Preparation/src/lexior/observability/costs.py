#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comptage des jetons et du coût des appels au modèle Teacher.

Les jetons sont LUS dans le champ `usage` renvoyé par l'API — ce ne sont pas
des estimations. Le coût, lui, est dérivé d'un tarif : il ne vaut que ce que
vaut le tarif configuré. Vérifiez-le sur votre facture.

Sans objet lorsque le Teacher est auto-hébergé (vLLM / Ollama) : le tarif est
alors nul et seuls les jetons sont comptés.

Utilisation :
    from api_cost import CostTracker
    tracker = CostTracker(model="gpt-4o-mini")
    ...
    response = client.chat.completions.create(...)
    tracker.record(response)
    ...
    tracker.report()
"""

import os
import threading

# Tarifs en USD par MILLION de jetons. Prix catalogue, à vérifier :
# un tarif obsolète donne un coût faux sans que rien ne le signale.
# Surchargeable par TEACHER_PRICE_IN / TEACHER_PRICE_OUT.
PRICING = {
    "gpt-4o-mini": {"in": 0.150, "cached_in": 0.075, "out": 0.600},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
}

# Modèle auto-hébergé : aucun coût à l'appel, on ne compte que les jetons.
SELF_HOSTED_HINTS = ("qwen", "llama", "mistral", "localhost", "vllm", "ollama")


class CostTracker:
    """Compteur thread-safe : generator_a2aj.py appelle l'API depuis N threads."""

    def __init__(self, model, price_in=None, price_out=None,
                 price_cached_in=None):
        self.model = model
        self.lock = threading.Lock()
        self.calls = 0
        self.failed = 0
        self.tokens_in = 0
        self.tokens_cached_in = 0
        self.tokens_out = 0

        key = (model or "").lower()
        if price_in is None:
            price_in = _env_float("TEACHER_PRICE_IN")
        if price_out is None:
            price_out = _env_float("TEACHER_PRICE_OUT")
        if price_cached_in is None:
            price_cached_in = _env_float("TEACHER_PRICE_CACHED_IN")

        if price_in is None or price_out is None:
            match = next((v for k, v in PRICING.items() if k in key), None)
            if match:
                price_in = price_in if price_in is not None else match["in"]
                price_out = price_out if price_out is not None else match["out"]
                price_cached_in = (price_cached_in if price_cached_in is not None
                                   else match.get("cached_in", price_in))
            elif any(h in key for h in SELF_HOSTED_HINTS):
                price_in, price_out = 0.0, 0.0     # auto-hébergé
                price_cached_in = 0.0
            else:
                price_in, price_out = 0.0, 0.0
                price_cached_in = 0.0
                self.unknown_pricing = True

        if price_cached_in is None:
            price_cached_in = price_in

        self.price_in = price_in
        self.price_cached_in = price_cached_in
        self.price_out = price_out
        self.unknown_pricing = getattr(self, "unknown_pricing", False)

    def record(self, response):
        """Comptabilise un appel réussi, à partir du champ usage de la réponse."""
        usage = getattr(response, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(prompt_details, "cached_tokens", 0) or 0
        cached = min(int(cached), int(pt))
        with self.lock:
            self.calls += 1
            self.tokens_in += pt
            self.tokens_cached_in += cached
            self.tokens_out += ct

    def record_failure(self):
        with self.lock:
            self.failed += 1

    @property
    def cost(self):
        uncached_in = self.tokens_in - self.tokens_cached_in
        return (uncached_in / 1e6) * self.price_in + \
               (self.tokens_cached_in / 1e6) * self.price_cached_in + \
               (self.tokens_out / 1e6) * self.price_out

    def snapshot(self):
        with self.lock:
            return {
                "model": self.model,
                "calls": self.calls,
                "failed_calls": self.failed,
                "tokens_in": self.tokens_in,
                "tokens_cached_in": self.tokens_cached_in,
                "tokens_out": self.tokens_out,
                "tokens_total": self.tokens_in + self.tokens_out,
                "price_in_per_1m_usd": self.price_in,
                "price_cached_in_per_1m_usd": self.price_cached_in,
                "price_out_per_1m_usd": self.price_out,
                "cost_usd": round(self.cost, 6),
            }

    def line(self):
        """Une ligne courte, pour un suivi en cours de génération."""
        return (f"{self.calls} appels | {self.tokens_in:,} in + {self.tokens_out:,} out "
                f"| {self.cost:.4f} USD cumulés")

    def report(self, label="Teacher"):
        s = self.snapshot()
        print(f"\n  --- COÛT DES APPELS ({label}) ---")
        print(f"  modèle                : {s['model']}")
        print(f"  appels réussis        : {s['calls']:,}")
        if s["failed_calls"]:
            print(f"  appels en échec       : {s['failed_calls']:,}  (non facturés ici)")
        print(f"  jetons entrée         : {s['tokens_in']:,}")
        if s["tokens_cached_in"]:
            print(f"    dont entrée en cache: {s['tokens_cached_in']:,}")
        print(f"  jetons sortie         : {s['tokens_out']:,}")
        print(f"  jetons total          : {s['tokens_total']:,}")
        if self.unknown_pricing:
            print(f"  tarif                 : INCONNU pour « {s['model']} » — coût non calculé.")
            print(f"                          Définir TEACHER_PRICE_IN / TEACHER_PRICE_OUT.")
        elif s["price_in_per_1m_usd"] == 0 and s["price_out_per_1m_usd"] == 0:
            print(f"  tarif                 : 0 (modèle auto-hébergé) — coût à l'appel nul.")
        else:
            print(f"  tarif                 : {s['price_in_per_1m_usd']} / "
                  f"{s['price_cached_in_per_1m_usd']} / "
                  f"{s['price_out_per_1m_usd']} USD par million "
                  f"(entrée / entrée en cache / sortie)")
            print(f"  COÛT                  : {s['cost_usd']:.4f} USD")
            if s["calls"]:
                print(f"  coût par ligne        : {s['cost_usd']/s['calls']:.6f} USD")
        return s


def _env_float(name):
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None
