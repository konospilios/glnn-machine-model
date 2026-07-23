# `glnn.py` — the GLNN implementation

~1370 lines of JAX implementing the General Lagrangian Neural Network: data loading
and differentiation, the two networks (`stax.serial`), the learned Lagrangian and
dissipation functions, the Euler–Lagrange `equation_of_motion`, the `loss`, and the
training loop.

## Provenance & caveats (please read)

- **This file was reconstructed from Appendix D of the project report**, because the
  original `.py` wasn't available at the time. The report embedded the code as a
  listing, so the source of truth here is that PDF listing — recovered as faithfully
  as possible, **for reading**.
- **How it was recovered (best-effort):** the PDF was extracted two ways — one pass
  for correct identifiers, one for indentation depth — then merged and re-indented
  using Python's own rules (a line ends in `:` → its block indents; brackets and
  triple-quoted strings tracked so their contents aren't misread). The result
  **passes `python -m py_compile`** (syntactically valid) and the line numbers match
  the report (e.g. the report states the equation of motion is on line 973 — and it
  is).
- **Known cosmetic artifacts** from the PDF listing: occasional spaces before commas
  (`stax.Softplus ,`), and closing brackets sometimes indented a level deeper than a
  human would write. These don't change behaviour.
- **Not runnable as-is:** it depends on JAX plus the (excluded, NDA) training data, so
  treat it as the method to read, not a turnkey script. If the original `.py` turns
  up, it should replace this reconstruction.
