# `glnn.py` — the GLNN implementation

~1370 lines of JAX. The main components:

- **Data pipeline** — loads the sensor signals and computes generalised velocities and
  accelerations by central-difference differentiation.
- **Two networks** (`stax.serial`) — one for the system's **Lagrangian**, one for the
  **dissipative (non-conservative) forces**.
- **`equation_of_motion`** — combines both networks through the **Euler–Lagrange
  equation** to predict accelerations.
- **`loss`** — mean-squared error between predicted and measured accelerations.
- **Training loop** — gradient-based optimisation over the two networks.
