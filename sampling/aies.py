import numpy as np
import tensorflow as tf

from .autocorr import integrated_time
from .base import BaseSampler


class AIESampler(BaseSampler):
    """
    Affine-invariant ensemble sampler (Goodman & Weare 2010) in TensorFlow
    """

    def __init__(self, n_walkers, ndim, log_prob_fn, a=2.0):
        if (isinstance(ndim, bool) or not isinstance(ndim, (int, np.integer)) or ndim < 1
                or isinstance(n_walkers, bool) or not isinstance(n_walkers, (int, np.integer))
                or n_walkers < 2 * ndim or not np.isfinite(a) or a <= 1):
            raise ValueError("AIES needs ndim >= 1, at least 2*ndim walkers, and finite a > 1")
        if n_walkers % 2 != 0:
            raise ValueError("Number of walkers must be even.")
        self.n_walkers = n_walkers
        self.ndim = ndim
        self.log_prob_fn = log_prob_fn
        self.a = a

        # Precompute constants used at every step to avoid repeated Python/Tensor allocations.
        self._half = n_walkers // 2
        self._ndim_m1 = tf.constant(ndim - 1, dtype=tf.float32)
        sqrt_a = float(a**0.5)

        # Stretch-move support mapped from u ~ U(0, 1): z in [1/a, a].
        self._z_offset = tf.constant(1.0 / sqrt_a, dtype=tf.float32)
        self._z_range = tf.constant(sqrt_a - 1.0 / sqrt_a, dtype=tf.float32)
        self._pos = None
        self._logp = None
        self._chain = None
        self._log_prob = None
        self._reset_state()

    def _reset_state(self):
        """Clear everything a run produces. Called from __init__, _initialize and reset
        so the three cannot drift apart as fields are added."""
        self._chain = None
        self._log_prob = None
        self._accept_count = None
        self._n_proposals = 0
        self._n_rows = 0
        self._max_tau = float("inf")

    def _initialize(self, initial_positions):
        positions = np.asarray(initial_positions, dtype=np.float32)
        if positions.shape != (self.n_walkers, self.ndim) or not np.isfinite(positions).all():
            raise ValueError("initial positions must be finite with shape (n_walkers, ndim)")
        centred = positions.astype(float) - positions.astype(float).mean(axis=0)
        scales = np.max(np.abs(centred), axis=0)
        if np.any(scales == 0) or np.linalg.matrix_rank(centred / scales) < self.ndim:
            raise ValueError("initial ensemble must span the full parameter space")
        self._pos = tf.convert_to_tensor(positions)
        self._logp = self.log_prob_fn(self._pos)
        lp = np.asarray(self._logp)
        if lp.shape != (self.n_walkers,) or np.isnan(lp).any() or np.isposinf(lp).any() or not np.isfinite(lp).any():
            raise ValueError("initial log probabilities must have walker shape and some finite support")
        self._reset_state()

    def _update_half(self, active_positions, active_log_probs, partner_pool):
        # z ~ g(z) ∝ 1/√z on [1/a, a]
        # Inverse CDF sampling gives
        # z = (u * (√a - 1/√a) + 1/√a)², where u ~ U(0, 1)
        u = tf.random.uniform((self._half,), dtype=tf.float32)
        z = tf.square(u * self._z_range + self._z_offset)
        partner_indices = tf.random.uniform(
            (self._half,), maxval=self._half, dtype=tf.int32
        )
        partner_positions = tf.gather(partner_pool, partner_indices)
        proposals = partner_positions + tf.expand_dims(z, axis=1) * (
            active_positions - partner_positions
        )
        proposal_log_probs = self.log_prob_fn(proposals)

        # r = z^(ndim-1) * proposal_prob / active_prob
        log_accept_ratio = (
            self._ndim_m1 * tf.math.log(z) + proposal_log_probs - active_log_probs
        )

        # a = min(1, r)
        accepted = (
            tf.math.log(tf.random.uniform((self._half,), dtype=tf.float32))
            < log_accept_ratio
        )
        new_positions = tf.where(
            tf.expand_dims(accepted, axis=1), proposals, active_positions
        )
        new_log_probs = tf.where(accepted, proposal_log_probs, active_log_probs)

        return new_positions, new_log_probs, accepted

    def _step(self, pos, logp):
        # Split-ensemble updates: each walker proposes against a complementary partner set.
        # Updating halves sequentially is the standard Goodman-Weare scheme.
        new_positions, new_log_probs, accepted = self._update_half(
            active_positions=pos[: self._half],
            active_log_probs=logp[: self._half],
            partner_pool=pos[self._half :],
        )

        # The second half conditions on the newly updated first half.
        new_positions2, new_log_probs2, accepted2 = self._update_half(
            active_positions=pos[self._half :],
            active_log_probs=logp[self._half :],
            partner_pool=new_positions,
        )

        return (
            tf.concat([new_positions, new_positions2], axis=0),
            tf.concat([new_log_probs, new_log_probs2], axis=0),
            tf.concat([accepted, accepted2], axis=0),
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _run_graph(self, n_steps, pos, logp):
        # TensorArray avoids Python-side appends and keeps storage on the TF side.
        chain = tf.TensorArray(
            dtype=tf.float32,
            size=n_steps,
            element_shape=tf.TensorShape([self.n_walkers, self.ndim]),
        )
        log_prob = tf.TensorArray(
            dtype=tf.float32,
            size=n_steps,
            element_shape=tf.TensorShape([self.n_walkers]),
        )

        accept_count = tf.zeros((self.n_walkers,), dtype=tf.int32)

        def cond(i, pos, logp, chain, log_prob, accept_count):
            return i < n_steps

        def body(i, pos, logp, chain, log_prob, accept_count):
            pos, logp, accepted = self._step(pos, logp)
            chain = chain.write(i, pos)
            log_prob = log_prob.write(i, logp)
            accept_count += tf.cast(accepted, tf.int32)
            return i + 1, pos, logp, chain, log_prob, accept_count

        _, pos, logp, chain, log_prob, accept_count = tf.while_loop(
            cond,
            body,
            loop_vars=[
                tf.constant(0, dtype=tf.int32),
                pos,
                logp,
                chain,
                log_prob,
                accept_count,
            ],
            # One logical MCMC iteration per loop body; keep execution order explicit.
            parallel_iterations=1,
        )

        return pos, logp, chain.stack(), log_prob.stack(), accept_count
    
    @tf.function(jit_compile=True, reduce_retracing=True)
    def _run_graph_no_storage(self, n_steps, pos, logp):
        def cond(i, pos, logp):
            return i < n_steps

        def body(i, pos, logp):
            pos, logp, _ = self._step(pos, logp)
            return i + 1, pos, logp

        _, pos, logp = tf.while_loop(
            cond,
            body,
            loop_vars=[
                tf.constant(0, dtype=tf.int32),
                pos,
                logp,
            ],
            parallel_iterations=1,
        )

        return pos, logp

    def run(self, n_steps, initial_positions=None, burn_in=0, progress=True,
            thin=1, chunk_size=5000):
        """Sample, storing the chain in a preallocated buffer.

        The chain used to be accumulated as a list of per-chunk tensors, concatenated
        with tf.concat and then copied again by .numpy(); all three lived at once, so the
        high-water mark was 2.9x one chain (measured: 1.13 GiB incremental for a 0.386 GiB
        16D chain). At 29D with the shipped 232 walkers x 100k steps that is 2.5 GiB per
        copy and ~7.3 GiB resident. Writing each chunk straight into a preallocated array
        and returning a view of the filled part costs exactly one copy.

        thin strides at write time rather than after the fact, so a thinned chain never
        allocates the unthinned one.

        The chain is always n_steps long. max(tau) is estimated after every run, so it can be
        checked how many autocorrelation times the chain holds.
        """
        if initial_positions is not None:
            self._initialize(initial_positions)
        if self._pos is None or self._logp is None:
            raise ValueError("Sampler not initialized. Provide initial_positions.")

        thin = max(1, int(thin))
        chunk_size = max(1, int(chunk_size))
        n_rows = (n_steps + thin - 1) // thin
        self._chain = np.empty((n_rows, self.n_walkers, self.ndim), dtype=np.float32)
        self._log_prob = np.empty((n_rows, self.n_walkers), dtype=np.float32)
        self._n_rows = 0

        total_accept_count = tf.zeros((self.n_walkers,), dtype=tf.int32)
        pos, logp = self._pos, self._logp

        if burn_in:
            pos, logp = self._run_graph_no_storage(
                tf.convert_to_tensor(burn_in, dtype=tf.int32), pos, logp
            )

        pbar = None
        if progress:
            from tqdm.auto import tqdm
            pbar = tqdm(total=n_steps, unit="step")

        total_steps = 0
        while total_steps < n_steps:
            steps = min(chunk_size, n_steps - total_steps)
            pos, logp, chain_chunk, logp_chunk, accept_count = self._run_graph(
                tf.convert_to_tensor(steps, dtype=tf.int32), pos, logp
            )
            total_accept_count += accept_count

            # Keep every thin-th step, counting from the start of the run so the stride
            # is unbroken across chunk boundaries.
            offset = (-total_steps) % thin
            if offset < steps:
                kept = chain_chunk[offset::thin].numpy()
                kept_logp = logp_chunk[offset::thin].numpy()
                end = self._n_rows + len(kept)
                self._chain[self._n_rows:end] = kept
                self._log_prob[self._n_rows:end] = kept_logp
                self._n_rows = end
            # Free the chunk before the next graph call rather than at rebinding: at
            # 5000 x 232 x 29 that tensor is 134 MB, and two of them need not coexist.
            del chain_chunk, logp_chunk

            total_steps += steps
            if pbar is not None:
                pbar.update(steps)

        if pbar is not None:
            pbar.close()

        if self._n_rows > 1:
            ac_thin = 10
            tau = integrated_time(self._chain[: self._n_rows : ac_thin],
                                  thin=thin * ac_thin)
            self._max_tau = float(np.max(tau))

        self._pos = pos
        self._logp = logp
        self._accept_count = total_accept_count
        self._n_proposals = total_steps

    # chain() and log_prob() return numpy views of the filled part of the buffer, not
    # tf tensors: the buffer is already numpy, and wrapping it back into a tensor for
    # the caller to .numpy() again would restore the copy this class exists to avoid.
    def chain(self, discard=0, thin=1):
        return self._chain[discard : self._n_rows : thin]

    def log_prob(self, discard=0, thin=1):
        return self._log_prob[discard : self._n_rows : thin]

    def last_positions(self):
        """The ensemble after the last step of the last run. Not chain()[-1] when the run was
        thinned: the stored rows are counted from the first step, so the last of them is up to
        thin - 1 steps before the end."""
        return self._pos.numpy()

    def acceptance_fraction(self):
        return tf.cast(self._accept_count, tf.float32) / tf.cast(
            self._n_proposals, tf.float32
        )

    @property
    def max_tau(self):
        return self._max_tau

    @property
    def n_steps_run(self):
        return self._n_proposals

    def reset(self):
        self._pos = None
        self._logp = None
        self._reset_state()
