"""Recurrent Temporal Restricted Boltzmann Machine (Bernoulli-Bernoulli).

DEV NOTE: this file lives in the sit_fuse_rtrbm dev package for now so it
can be built/tested against the real installed `learnergy` package without
naming collisions. Once ready for the actual upstream contribution, this
file gets copied into a fork of github.com/gugarosa/learnergy at
learnergy/models/temporal/rtrbm.py -- see Technical Design Note.

Reference:
    I. Sutskever, G. Hinton, G. Taylor. The recurrent temporal restricted
    Boltzmann machine. NeurIPS (2008).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import learnergy.utils.constants as c
import learnergy.utils.exception as e
from learnergy.models.bernoulli import RBM
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class RTRBM(RBM):
    """Single-layer Bernoulli-Bernoulli RTRBM. Extends RBM (learnergy's
    bernoulli/rbm.py) with a hidden-to-hidden recurrent weight matrix W'
    and a learnable initial hidden state, so the hidden bias at each
    timestep is conditioned on the mean-field hidden probabilities of the
    PREVIOUS timestep.
    """

    def __init__(
        self,
        n_visible: int = 128,
        n_hidden: int = 128,
        steps: int = 1,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        decay: float = 0.0,
        temperature: float = 1.0,
        use_gpu: bool = False,
    ) -> None:
        """Initialization method.

        Args mirror RBM's __init__ exactly (see learnergy's rbm.py) -- no
        new hyperparameters at the base layer beyond what's needed for the
        recurrent connection, added below.
        """

        logger.info("Overriding class: RBM -> RTRBM.")

        super(RTRBM, self).__init__(
            n_visible, n_hidden, steps, learning_rate, momentum,
            decay, temperature, use_gpu,
        )

        # Recurrent hidden-to-hidden weights: W' in the paper.
        self.W_prime = nn.Parameter(torch.randn(n_hidden, n_hidden) * 0.01)

        # Learnable initial hidden state, used at t=0 (no h_{-1} exists).
        self.h0 = nn.Parameter(torch.zeros(n_hidden))

        # Mirrors VarianceGaussianRBM's pattern (gaussian_rbm.py) of
        # registering new nn.Parameters with the optimizer AFTER
        # super().__init__() has already built it.
        self.optimizer.add_param_group({"params": [self.W_prime, self.h0]})

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")

    @property
    def W_prime(self) -> torch.nn.Parameter:
        """Recurrent hidden-to-hidden weights matrix."""
        return self._W_prime

    @W_prime.setter
    def W_prime(self, W_prime: torch.nn.Parameter) -> None:
        self._W_prime = W_prime

    @property
    def h0(self) -> torch.nn.Parameter:
        """Learnable initial hidden state (used at the first timestep)."""
        return self._h0

    @h0.setter
    def h0(self, h0: torch.nn.Parameter) -> None:
        self._h0 = h0

    def hidden_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor, scale: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h_t | v_t, h_{t-1}).

        Overrides RBM.hidden_sampling (learnergy's rbm.py) to add the
        recurrent term. NOTE the changed signature vs. the parent class --
        this now REQUIRES h_prev (previous timestep's mean-field hidden
        probs). Deliberate deviation from RBM's API -- see RTRBM Technical
        Design Note re: forward-compatibility with DBN.forward().

        Args:
            v: Visible layer tensor for the CURRENT timestep, shape
                (batch, n_visible).
            h_prev: Mean-field hidden probabilities from the PREVIOUS
                timestep, shape (batch, n_hidden). Use self.h0 (broadcast
                to batch size) for the first timestep in a sequence.
            scale: same role as in RBM.hidden_sampling -- whether to divide
                by temperature T.

        Returns:
            Probabilities and states of the hidden layer sampling.
        """
        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(v, self.W.t()) + recurrent_bias

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def energy(self, samples: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Calculates and frees the system's energy, ACCOUNTING FOR the
        recurrent bias term.

        Overrides RBM.energy (learnergy's rbm.py). NECESSARY override, not
        optional: the parent class's energy() uses only the plain bias
        `self.b`, which would silently ignore the recurrent contribution
        `W_prime @ h_prev` that hidden_sampling actually uses. If left
        un-overridden, the energy used to compute the CD-k training cost
        would not match the distribution actually being sampled from --
        meaning W_prime's gradient signal would be wrong/missing during
        training, even though hidden_sampling itself works correctly.

        Args:
            samples: Samples to be energy-freed, shape (batch, n_visible).
            h_prev: Same previous-timestep hidden probabilities used in
                hidden_sampling for this timestep, shape (batch, n_hidden).

        Returns:
            The system's energy based on input samples, shape (batch,).
        """
        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(samples, self.W.t()) + recurrent_bias

        s = nn.Softplus()
        h = torch.sum(s(activations), dim=1)
        v = torch.mv(samples, self.a)

        energy = -v - h

        return energy

    def gibbs_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the whole Gibbs sampling procedure FOR ONE TIMESTEP.

        Overrides RBM.gibbs_sampling (learnergy's rbm.py) to thread h_prev
        through every hidden_sampling call. IMPORTANT: h_prev stays FIXED
        across all CD-k steps within this call -- it is the recurrent
        context for THIS timestep only, not something that changes during
        the k Gibbs-sampling bounces. (The recurrence across TIMESTEPS
        happens one level up, in the training loop that calls this
        function once per timestep with an updated h_prev each time --
        not implemented yet, see fit() below.)

        This matches "independent CD-k per timestep" as decided with Nick
        (2026-06-29): each timestep runs its own self-contained CD-k, with
        the only cross-timestep influence being the fixed h_prev bias.

        Args:
            v: Visible layer tensor for the CURRENT timestep.
            h_prev: Previous timestep's hidden probabilities (fixed for
                the duration of this call).

        Returns:
            Same 5-tuple as RBM.gibbs_sampling: positive hidden
            probs/states, negative hidden probs/states, negative visible
            states.
        """
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v, h_prev)
        neg_hidden_states = pos_hidden_states

        for _ in range(self.steps):
            _, visible_states = self.visible_sampling(neg_hidden_states, True)
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, h_prev, True
            )

        return (
            pos_hidden_probs,
            pos_hidden_states,
            neg_hidden_probs,
            neg_hidden_states,
            visible_states,
        )

    def cd_step(self, v: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Performs ONE Contrastive Divergence training step, for ONE
        timestep's worth of data. This is the basic building block that
        fit() will eventually loop over -- one frame, one CD-k update.

        Mirrors the inner-loop body of RBM.fit() (learnergy's rbm.py)
        almost exactly -- positive phase -> Gibbs sampling -> cost ->
        backward -> optimizer step -- just using THIS class's
        gibbs_sampling/energy (which both take h_prev) instead of the
        parent's.

        NOTE on pseudo_likelihood: RBM.fit() also tracks log-PL via
        self.pseudo_likelihood(), but that inherited method internally
        calls self.energy(samples) with the PARENT's one-argument
        signature -- incompatible with RTRBM's energy(samples, h_prev).
        This is consistent with the Technical Design Note's existing
        deferral of pseudo_likelihood for this model. Not computed here;
        only MSE is tracked for now.

        NOTE: this does NOT zero/step the optimizer's gradient
        accumulation across multiple calls in a sequence -- that
        coordination (zero_grad once per subseries, not once per
        timestep, so gradients accumulate correctly across the whole
        chain for BPTT) belongs in fit()'s outer loop, not here. Calling
        this method standalone (as the smoke test below does) performs
        zero_grad/step every call, which is correct ONLY for testing a
        single isolated timestep in isolation -- not for real subseries
        training.

        Args:
            v: Visible layer tensor for this timestep, shape
                (batch, n_visible).
            h_prev: Previous timestep's hidden probabilities, shape
                (batch, n_hidden).

        Returns:
            mse -- reconstruction error for this one step.
        """
        _, _, _, _, visible_states = self.gibbs_sampling(v, h_prev)
        visible_states = visible_states.detach()

        cost = torch.mean(self.energy(v, h_prev)) - torch.mean(
            self.energy(visible_states, h_prev)
        )

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        batch_size = v.size(0)
        mse = torch.div(
            torch.sum(torch.pow(v - visible_states, 2)), batch_size
        ).detach()

        return mse

    def fit_subseries(self, sequence: torch.Tensor) -> torch.Tensor:
        """Trains on ONE subseries (a short chunk of consecutive
        timesteps), using independent CD-k per timestep for the cost
        computation, but accumulating cost ACROSS THE WHOLE SUBSERIES
        before a SINGLE backward() + optimizer step.

        This is the actual BPTT mechanism: h_prev is carried forward
        WITHOUT detaching between timesteps, so gradients from later
        timesteps' cost can flow backward through the h_prev chain into
        earlier timesteps' contribution to W_prime and h0. This matches
        Nick's decision (2026-06-29): train on subseries (not full long
        sequences), independent CD-k per timestep for the W/a/b
        contribution, per the original paper.

        Contrast with cd_step(): that method does zero_grad/backward/step
        on EVERY call, which is correct for testing one timestep in total
        isolation, but WRONG for chained training -- it would correct the
        model after every single frame and never let gradients flow
        across the chain at all. This method is the one to actually use
        for subseries training.

        Args:
            sequence: ONE subseries, shape (batch, seq_len, n_visible).

        Returns:
            mse -- summed reconstruction error across all timesteps in
            the subseries (a single scalar, already detached).
        """
        batch_size, seq_len, n_visible = sequence.shape

        h_prev = self.h0.unsqueeze(0).expand(batch_size, -1)

        self.optimizer.zero_grad()

        total_cost = torch.tensor(0.0)
        total_mse = torch.tensor(0.0)

        for t in range(seq_len):
            v_t = sequence[:, t, :]

            _, _, _, _, visible_states = self.gibbs_sampling(v_t, h_prev)
            visible_states = visible_states.detach()

            cost_t = torch.mean(self.energy(v_t, h_prev)) - torch.mean(
                self.energy(visible_states, h_prev)
            )
            total_cost = total_cost + cost_t

            batch_mse = torch.div(
                torch.sum(torch.pow(v_t - visible_states, 2)), batch_size
            ).detach()
            total_mse = total_mse + batch_mse

            # CRITICAL: do NOT detach here. h_prev for the NEXT timestep
            # must stay attached to the computation graph, or gradients
            # from this timestep onward could never flow back to W_prime/
            # h0's contribution at EARLIER timesteps -- which would
            # silently turn this into "independent CD-k with no real BPTT
            # at all," defeating the entire point of this method vs.
            # cd_step().
            h_prev, _ = self.hidden_sampling(v_t, h_prev)

        total_cost.backward()
        self.optimizer.step()

        return total_mse

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> torch.Tensor:
        """Fits a new RTRBM model on a dataset of subseries sequences.

        Outer training loop wrapping fit_subseries() -- mirrors RBM.fit()
        (learnergy's rbm.py) in structure: DataLoader -> epoch loop ->
        batch loop -> per-epoch self.dump() for history tracking.

        Key differences from RBM.fit():
        - Each "sample" is a SEQUENCE of shape (seq_len, n_visible), not
          a flat (n_visible,) vector. DataLoader returns batches of shape
          (batch, seq_len, n_visible) -- no reshaping needed.
        - Inner training step is fit_subseries(), not a single CD-k call.
        - Returns only mse (not (mse, pl)) since pseudo_likelihood is
          deferred for this model (see cd_step() docstring).

        Args:
            dataset: A SFTemporalDataset (or compatible Dataset) where
                each sample is a sequence of shape (seq_len, n_visible).
            batch_size: Number of sequences per batch.
            epochs: Number of full passes through the dataset.

        Returns:
            mse from the final epoch.
        """
        import time
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        mse = torch.tensor(0.0)

        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            start = time.time()
            mse = torch.tensor(0.0)

            for samples, _ in tqdm(batches):
                # samples shape: (batch, seq_len, n_visible)
                # No reshape needed -- fit_subseries expects this shape.
                if self.device == "cuda":
                    samples = samples.cuda()

                batch_mse = self.fit_subseries(samples)
                mse += batch_mse

            mse /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), time=end - start)

            logger.info("MSE: %f", mse)

        return mse