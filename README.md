# Inferring Earth’s Dynamics from Partial Solar System Observations

This repository studies whether machine learning models can recover the Earth’s motion and physical parameters using only indirect observations of other celestial bodies.  
The project combines **Physics-Informed Neural Networks (PINNs)** and **data-driven sequence models (LSTMs)** to address dynamics recovery and forecasting under partial observability.

---

## Repository Structure

### `functions.py`
This file contains all core models and utilities used throughout the experiments:

- **Data utilities**
  - `prepare_data`: prepares input–output pairs for partial-observation PINN training.

- **Physics-Informed Neural Networks**
  - `PINN_NBody`: PINN for recovering the full Solar System state when gravitational constant and masses are known.
  - `PINN_NBody_unk`: extended PINN that learns the gravitational constant and planetary masses directly from data (inverse problem).

  Both PINN models enforce Newtonian $N$-body equations via a physics loss.

- **Sequence-to-Sequence LSTM**
  - `Seq2SeqLSTM`: encoder–decoder LSTM with autoregressive decoding and optional teacher forcing for time-series forecasting.

All experiments import models directly from this file.

---

### Notebooks

- **`solar_system_equations.ipynb`**  
  Implements and verifies Newtonian $N$-body equations and validates the physical consistency of the dataset.

- **`solar_system_pinn.ipynb`**  
  Trains and evaluates PINN models for:
  - Earth state reconstruction from partial observations,
  - Inverse learning of gravitational constant and planetary masses,
  - Generalization under physics-only supervision.

- **`solar_system_lstm.ipynb`**  
  Trains sequence-to-sequence LSTM models for Earth trajectory prediction:
  - Using all bodies’ past states,
  - Using Earth-only historical data.

---

### Simulation Code

- **`solar_system.c`**  
  C implementation of the Solar System simulation using the REBOUND N-body integrator.
  Generates high-precision trajectory data (`state.csv`) used by all learning models.

---

## Summary

- PINNs enforce Newtonian dynamics to recover hidden states and physical parameters.
- LSTMs provide data-driven baselines for short-term forecasting.
- All models and losses are defined in `functions.py`; notebooks focus on experimentation and analysis.

---

## Notes

- The `oscillator_sindy.ipynb` notebook is unrelated to the Solar System experiments and can be ignored.
