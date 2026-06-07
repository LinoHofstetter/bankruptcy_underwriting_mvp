# Historical Company Risk Underwriting

We are building a **company-risk intelligence layer for commercial underwriting**.  
The system estimates future adverse-event risk for a company using financials, filings, documents, news, relationships, and insurance-product context.  
The goal is to help insurers identify high-risk companies earlier, review portfolios faster, and make underwriting decisions with better evidence.

---

## Vision

Commercial underwriting is still heavily manual, fragmented, and document-driven.  
Underwriters need to understand not only a company's financial position, but also its legal, operational, cyber, regulatory, and reputational risk.

Our model should answer:

> Given a company, an underwriting date, and an insurance product, what is the probability of a relevant adverse event within the next prediction horizon?

Relevant events depend on the product:

| Product | Example Risk Events |
| --- | --- |
| Trade credit | bankruptcy, default, payment failure |
| Cyber | breach, ransomware, IT outage |
| D&O | litigation, regulatory action, restatement |
| Workers' comp | workplace incident, safety violation |
| General liability | lawsuit, product liability event |

---

## Target Architecture

The long-term system is a **multimodal, product-conditioned risk model**.

```text
company financials
+ filing history
+ company documents
+ news and event timelines
+ company relationship graph
+ insurance product context
        ↓
point-in-time company snapshot
        ↓
cross-attention risk model
        ↓
calibrated risk probability
        ↓
risk drivers + supporting evidence
```

The key idea is that the insurance product should determine which company signals matter.  
A cyber policy should focus on breach history, IT outages, and cyber-risk disclosures.  
A D&O policy should focus on litigation, governance, executive turnover, and regulatory events.

This can be modeled with cross-attention:

```text
insurance product tokens
        ↓
attend over company evidence tokens
        ↓
product-specific risk representation
        ↓
risk score + referral recommendation + explanation
```

A critical requirement is **point-in-time correctness**: the model may only use information that was available at the underwriting date. This prevents leakage and makes historical evaluation realistic.

---

## Current Prototype

The current prototype demonstrates the core workflow on a clean proxy task:

```text
historical financial ratios → calibrated one-year bankruptcy risk
```

It uses the UCI Polish Companies Bankruptcy dataset with a CatBoost model, probability calibration, risk ranking, simple explanations, and a Streamlit inspection app.

This prototype is not the final underwriting model. It is a minimal demonstration that the end-to-end system can load company data, train a risk model, calibrate probabilities, rank companies, explain predictions, and support interactive review.
