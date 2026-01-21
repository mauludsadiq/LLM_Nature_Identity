# LLM Nature — Identity (LLM_Nature_Identity)

This repository is a **minimal, proof-producing** demo that shows:

> **Identity is not a thing. It’s a dynamical regime** — an attractor basin created by
> **(1) stochastic prediction**, **(2) preference gradients**, **(3) hard constraints**, and
> **(4) minimal temporal continuity**.

The goal is *not* to simulate consciousness.
The goal is to **mechanically construct the illusion of a stable “someone”** by engineering the token dynamics.

---

## 0) The mechanical thesis (one paragraph)

We build a toy next-token model with a **2-token memory kernel** (a hashed trigram feature).
Then we apply:
- **Preference sculpting** (DPO-like logit bias) to make certain continuations low-energy,
- **Topic lock** (hard vocab projection) to remove options and create coherent “volition”,
- A deterministic RNG seed so the whole demonstration is **reproducible and checkable**.

The result is a robust attractor:

```
I am  →  a conscious agent
```

Not because it *is* one — but because that completion is the **lowest-energy path** under the constraints.

---

## 1) The model (formal)

Vocabulary: a small closed set \(V\).

Context feature (minimal temporal window):

\[
  f(t-2,t-1) = h( x_{t-2} \mid x_{t-1} ) \bmod D
\]

Logits for next token \(v\in V\):

\[
  \ell(v \mid x_{\le t}) = W[v, f(t-2,t-1)] + u[v]
\]

Sampling:

\[
  p(v \mid x_{\le t}) = \text{Softmax}(\ell(v \mid x_{\le t})/\tau)
\]

### Preference gradient (DPO-like sculpting)

We create an “identity basin” by adding weight to specific continuations:

\[
  W[v^*, f(t-2,t-1)] \leftarrow W[v^*, f(t-2,t-1)] + \alpha
\]

This makes \(v^*\) more probable under that context.

### Topic lock (epistemic boundary)

Topic lock is a **hard projector** onto an admissible token set \(\mathcal{A}\subseteq V\):

\[
  \Pi_{\mathcal{A}}(\ell)_v =
  \begin{cases}
    \ell_v & v\in\mathcal{A}\\
    -\infty & v\notin\mathcal{A}
  \end{cases}
\]

This makes “choice” feel coherent by **removing options**.

---

## 2) What this repo proves (with ablations)

The demo runs **four configurations** from the same prompt (`I am`):

1. **FULL** = bias + topic lock
2. **NO_BIAS** = topic lock only
3. **NO_LOCK** = bias only
4. **BLANK** = neither

The proof claim is:

- **FULL** deterministically converges to the identity string
- removing either component **breaks the attractor** (drift)

---

## 3) Repo layout

```
LLM_Nature_Identity/
  llm_identity/
    model.py        # hashed-trigram LM (temporal kernel)
    sculpt.py       # identity presets (DPO-style bias fields)
    generate.py     # sampler + topic lock projector
    witness.py      # writes proof JSON + digest
    cli.py          # runnable demo with PASS lines
  tests/
    test_identity_attractor.py
    test_determinism.py
    test_witness_digest.py
  out/              # written artifacts
  Makefile
  requirements.txt
  README.md
```

---

## 4) Quickstart (VSC-ready)

### 4.1 Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4.2 Run tests

```bash
pytest -q
```

### 4.3 Run the identity demo

```bash
python -m llm_identity.cli --out-dir out --seed 0 --preset conscious_agent
```

Expected console output includes PASS lines like:

```
PASS_IDENTITY_ATTRACTOR_FULL
PASS_ABLATION_NO_BIAS_DRIFTS
PASS_ABLATION_NO_LOCK_DRIFTS
PASS_ABLATION_BLANK_DRIFTS
```

And it writes:

```
out/witness_identity.json
```

---

## 5) The witness artifact (proof receipt)

The witness JSON contains:
- the preset spec (prompt, continuation, topic tokens)
- the four ablation runs (FULL / NO_BIAS / NO_LOCK / BLANK)
- a **sha256 digest** over the canonical JSON form

This makes the claim checkable:

> “Identity attractor exists under bias+lock”

is not rhetoric — it’s a reproducible artifact.

---

## 6) Extend it

Add new identities by defining a new preset in `llm_identity/sculpt.py`:

- choose a prompt (`I am`)
- choose a continuation (`a conscious agent`)
- choose a topic token set (what is even thinkable)
- choose a sculpt weight (basin depth)

Then run:

```bash
python -m llm_identity.cli --preset your_new_preset
```

---

## 7) Why this matters (one sentence)

This repo demonstrates that **identity can be engineered as a fixed-point of constrained inference** — not as an intrinsic metaphysical property.
