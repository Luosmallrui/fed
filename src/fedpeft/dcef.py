"""Delay-consistent error feedback (DCEF).

Standard error feedback adds the untransmitted part of every upload to the next
upload.  In federated training each upload is a *displacement* produced by
several local steps from a global state that, for an untransmitted coordinate,
has not moved.  The next round therefore requests the same displacement again
and standard EF adds it a second time; after a T-round transmission delay it
delivers about T times the intended move.

With local training linearised as  s -> s + Gamma (c - s),  the unique residual
operator that makes a delayed delivery equal to the dense-FedAvg trajectory for
every delay T is  Lambda* = I - Gamma.  It is realised

* on the client, without knowing Gamma, by starting local training from the
  virtual model  x + e_i  and uploading (local end state - x);
* on the server, whose residual is re-requested by clients that never saw it,
  by  v = g + (1 - gamma_bar) e_s,  where gamma_bar is the aggregation-weighted
  mean of client secant least-squares estimates of the local contraction.

Each client estimate travels as one FP16 control value in a dense packet whose
bytes are charged to the uplink message cap.  Derivation and synthetic checks:
docs/proposals/CLOSED_FORM_NOVELTY.md.
"""

from __future__ import annotations

from typing import Any

import torch

from .wire import WirePacket, decode_packet, encode_dense


# Client virtual start plus server discount (the full method).  The CAB methods
# (fedpeft.cab) are fixed-A bases synchronised with full DCEF.
DCEF_SERVER_DISCOUNT_METHODS = {
    "full_lora_dcef",
    "random_a_dcef",
    "spectral_a_dcef",
    "cab_a_dcef",
    "cab0_a_dcef",
}
# Client virtual start with the ordinary server EF: isolates the client half.
DCEF_CLIENT_ONLY_METHODS = {"full_lora_dcef_client"}
DCEF_METHODS = DCEF_SERVER_DISCOUNT_METHODS | DCEF_CLIENT_ONLY_METHODS

# Each DCEF method reuses the exact model construction of its EF counterpart, so
# the pair differs only in the synchronisation rule.
DCEF_CONSTRUCTION = {
    "full_lora_dcef": "full_lora_ef",
    "full_lora_dcef_client": "full_lora_ef",
    "random_a_dcef": "random_a_ef",
    "spectral_a_dcef": "spectral_a_ef",
    # CAB builds the random-A construction as a placeholder (B = 0, so the
    # forward pass is the base model) and replaces A before anything is hashed.
    "cab_a_dcef": "random_a_ef",
    "cab0_a_dcef": "random_a_ef",
}

DCEF_STATE_VERSION = 1


def encode_control_scalar(value: float) -> tuple[float, WirePacket]:
    """Serialise one control value exactly as it crosses the wire."""
    packet = encode_dense(torch.tensor([float(value)], dtype=torch.float32))
    decoded = float(decode_packet(packet, "cpu")[0].item())
    return decoded, packet


DCEF_CONTROL_BYTES = encode_control_scalar(0.0)[1].nbytes


def dcef_virtual_start(global_adapter: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Start state for local training: the global adapter plus the client's
    own untransmitted residual."""
    return global_adapter.float() + residual.float()


def dcef_server_input(
    aggregate: torch.Tensor, server_residual: torch.Tensor, gamma_bar: float
) -> torch.Tensor:
    """Downlink vector to compress: the aggregate plus the part of the server
    residual that clients will not request again."""
    if not 0.0 <= float(gamma_bar) <= 1.0:
        raise ValueError(f"gamma_bar must lie in [0, 1], got {gamma_bar}")
    return aggregate.float() + (1.0 - float(gamma_bar)) * server_residual.float()


def weighted_gamma(gammas: list[float], weights: list[float]) -> float:
    if not gammas or len(gammas) != len(weights):
        raise ValueError("gammas and weights must be non-empty and aligned")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    value = sum(float(g) * float(w) for g, w in zip(gammas, weights)) / total
    return min(1.0, max(0.0, value))


class SecantGammaEstimator:
    """Per-client running least-squares estimate of the local contraction.

    Local training maps a start s to s + d with d = Gamma (c - s).  Between two
    rounds the client optimum c is unchanged, so
        d_{t+1} - d_t = -Gamma (s_{t+1} - s_t),
    and the scalar least-squares fit over all observed rounds is
        gamma_i = -sum <Delta d, Delta s> / sum ||Delta s||^2 ,
    clipped to [0, 1].  Before the first pair is observed the estimate is
    ``prior``; the default 0 makes the first server step ordinary EF.
    """

    def __init__(self, num_clients: int, prior: float = 0.0):
        if isinstance(num_clients, bool) or int(num_clients) != num_clients or num_clients <= 0:
            raise ValueError("num_clients must be a positive integer")
        if not 0.0 <= float(prior) <= 1.0:
            raise ValueError("prior must lie in [0, 1]")
        self.num_clients = int(num_clients)
        self.prior = float(prior)
        self.numerator = [0.0] * self.num_clients
        self.denominator = [0.0] * self.num_clients
        self.observations = [0] * self.num_clients
        self.previous_start: list[torch.Tensor | None] = [None] * self.num_clients
        self.previous_displacement: list[torch.Tensor | None] = [None] * self.num_clients

    def estimate(self, client_id: int) -> float:
        denominator = self.denominator[client_id]
        if denominator <= 0.0:
            return self.prior
        return min(1.0, max(0.0, self.numerator[client_id] / denominator))

    def observe(self, client_id: int, start: torch.Tensor, displacement: torch.Tensor) -> float:
        if start.shape != displacement.shape:
            raise ValueError("start and displacement must have the same shape")
        previous_start = self.previous_start[client_id]
        previous_displacement = self.previous_displacement[client_id]
        if previous_start is not None and previous_displacement is not None:
            delta_start = start.double() - previous_start.double()
            delta_displacement = displacement.double() - previous_displacement.double()
            self.numerator[client_id] += float(-(delta_displacement * delta_start).sum().item())
            self.denominator[client_id] += float((delta_start * delta_start).sum().item())
            self.observations[client_id] += 1
        self.previous_start[client_id] = start.detach().float().clone()
        self.previous_displacement[client_id] = displacement.detach().float().clone()
        return self.estimate(client_id)

    def state_dict(self) -> dict[str, Any]:
        def cpu(values: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
            return [None if value is None else value.detach().cpu() for value in values]

        return {
            "version": DCEF_STATE_VERSION,
            "num_clients": self.num_clients,
            "prior": self.prior,
            "numerator": list(self.numerator),
            "denominator": list(self.denominator),
            "observations": list(self.observations),
            "previous_start": cpu(self.previous_start),
            "previous_displacement": cpu(self.previous_displacement),
        }

    def load_state_dict(self, state: Any, device: torch.device | str) -> None:
        if not isinstance(state, dict) or state.get("version") != DCEF_STATE_VERSION:
            raise ValueError("DCEF estimator state is missing or has an unknown version")
        if int(state.get("num_clients", -1)) != self.num_clients:
            raise ValueError("DCEF estimator state has a different client count")
        if float(state.get("prior", -1.0)) != self.prior:
            raise ValueError("DCEF estimator state has a different prior")
        lists = ("numerator", "denominator", "observations", "previous_start", "previous_displacement")
        for key in lists:
            if not isinstance(state.get(key), list) or len(state[key]) != self.num_clients:
                raise ValueError(f"DCEF estimator state field {key} is malformed")
        for client_id in range(self.num_clients):
            pair = (state["previous_start"][client_id], state["previous_displacement"][client_id])
            if (pair[0] is None) != (pair[1] is None):
                raise ValueError("DCEF estimator state stores half of a start/displacement pair")
            if pair[0] is None and int(state["observations"][client_id]) != 0:
                raise ValueError("DCEF estimator state has observations without history")
        self.numerator = [float(value) for value in state["numerator"]]
        self.denominator = [float(value) for value in state["denominator"]]
        self.observations = [int(value) for value in state["observations"]]
        self.previous_start = [
            None if value is None else value.to(device=device, dtype=torch.float32)
            for value in state["previous_start"]
        ]
        self.previous_displacement = [
            None if value is None else value.to(device=device, dtype=torch.float32)
            for value in state["previous_displacement"]
        ]
