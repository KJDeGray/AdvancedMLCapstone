import numpy as np

from OnlineActorCriticRSTDP import OnlineActorCriticRSTDP


class OnlineActorCriticLSM(OnlineActorCriticRSTDP):
    """
    Liquid-state/reservoir spiking actor-critic.

    This extends the R-STDP actor-critic with a fixed recurrent hidden
    reservoir. The recurrent synapses are not trained; actor, critic, input,
    and biases continue to use the same three-factor update as
    OnlineActorCriticRSTDP. This makes it closer in spirit to DQN-RC/ESN:
    rich temporal dynamics, simple readout-style learning.
    """

    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.002,
        reward_decay=0.9,
        e_greedy=0.1,
        hidden_size=64,
        actor_pop_size=6,
        critic_pop_size=8,
        spike_window=30,
        mem_decay=0.92,
        trace_decay=0.88,
        eligibility_decay=0.95,
        threshold=3.0,
        input_scale=1.0,
        value_scale=6.0,
        actor_output_gain=10.0,
        recurrent_scale=0.4,
        recurrent_sparsity=0.9,
        weight_clip=2.0,
        target_spike_count=900.0,
        homeostatic_lr=0.002,
        weight_decay=0.0001,
        actor_bias_modulation=0.3,
        actor_lr=None,
        critic_lr=None,
        input_lr=None,
        seed=None,
    ):
        self.recurrent_scale = float(recurrent_scale)
        self.recurrent_sparsity = float(recurrent_sparsity)
        super().__init__(
            n_actions=n_actions,
            n_features=n_features,
            learning_rate=learning_rate,
            reward_decay=reward_decay,
            e_greedy=e_greedy,
            hidden_size=hidden_size,
            actor_pop_size=actor_pop_size,
            critic_pop_size=critic_pop_size,
            spike_window=spike_window,
            mem_decay=mem_decay,
            trace_decay=trace_decay,
            eligibility_decay=eligibility_decay,
            threshold=threshold,
            input_scale=input_scale,
            value_scale=value_scale,
            actor_output_gain=actor_output_gain,
            weight_clip=weight_clip,
            target_spike_count=target_spike_count,
            homeostatic_lr=homeostatic_lr,
            weight_decay=weight_decay,
            actor_bias_modulation=actor_bias_modulation,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            input_lr=input_lr,
            seed=seed,
        )
        self.W_rec = self._make_reservoir()
        self.prev_h_spikes = np.zeros(self.hidden_size)

    def _make_reservoir(self):
        W = self.rng.randn(self.hidden_size, self.hidden_size) / np.sqrt(max(self.hidden_size, 1))
        mask = self.rng.rand(self.hidden_size, self.hidden_size) > self.recurrent_sparsity
        W *= mask
        np.fill_diagonal(W, 0.0)

        radius = np.max(np.abs(np.linalg.eigvals(W))) if np.any(W) else 0.0
        if radius > 1e-12:
            W *= self.recurrent_scale / radius
        return W

    def reset_state(self):
        super().reset_state()
        self.prev_h_spikes = np.zeros(self.hidden_size)

    def _state_snapshot(self):
        return super()._state_snapshot() + (self.prev_h_spikes.copy(),)

    def _restore_snapshot(self, snapshot):
        super()._restore_snapshot(snapshot[:-1])
        self.prev_h_spikes = snapshot[-1].copy()

    def _forward(self, state, commit=True):
        snapshot = None if commit else self._state_snapshot()

        x_spikes = self._rate_encode(state)
        h_counts = np.zeros(self.hidden_size)
        actor_counts = np.zeros(self.n_actor_neurons)
        critic_counts = np.zeros(self.critic_pop_size)

        step_e_in = np.zeros_like(self.e_in)
        step_e_actor = np.zeros_like(self.e_actor)
        step_e_critic = np.zeros_like(self.e_critic)

        for x_t in x_spikes:
            recurrent_current = self.W_rec @ self.prev_h_spikes
            h_current = self.W_in @ x_t + recurrent_current + self.b_h
            self.v_h, h_spikes = self._lif_step(self.v_h, h_current)
            self.prev_h_spikes = h_spikes

            actor_current = self.W_actor @ h_spikes + self.b_actor
            self.v_actor, actor_spikes = self._lif_step(self.v_actor, actor_current)

            critic_current = self.W_critic @ h_spikes + self.b_critic
            self.v_critic, critic_spikes = self._lif_step(self.v_critic, critic_current)

            self.x_trace = self.trace_decay * self.x_trace + x_t
            self.h_trace = self.trace_decay * self.h_trace + h_spikes
            self.actor_trace = self.trace_decay * self.actor_trace + actor_spikes
            self.critic_trace = self.trace_decay * self.critic_trace + critic_spikes

            local_e_in = np.outer(h_spikes, self.x_trace)
            local_e_actor = np.outer(actor_spikes, self.h_trace)
            local_e_critic = np.outer(critic_spikes, self.h_trace)

            self.e_in = self.eligibility_decay * self.e_in + local_e_in
            self.e_actor = self.eligibility_decay * self.e_actor + local_e_actor
            self.e_critic = self.eligibility_decay * self.e_critic + local_e_critic

            step_e_in += self.e_in
            step_e_actor += self.e_actor
            step_e_critic += self.e_critic

            h_counts += h_spikes
            actor_counts += actor_spikes
            critic_counts += critic_spikes

        action_rates = self._decode_actor(actor_counts / max(self.spike_window, 1))
        probs = self._softmax(self.actor_output_gain * action_rates)
        critic_rate = float(np.mean(critic_counts) / max(self.spike_window, 1))
        value = self.value_scale * critic_rate
        spike_count = float(np.sum(x_spikes) + np.sum(h_counts) + np.sum(actor_counts) + np.sum(critic_counts))

        out = {
            "probs": probs,
            "action_rates": action_rates,
            "value": value,
            "spike_count": spike_count,
            "h_rate": h_counts / max(self.spike_window, 1),
            "actor_rate_neurons": actor_counts / max(self.spike_window, 1),
            "critic_rate_neurons": critic_counts / max(self.spike_window, 1),
            "e_in": step_e_in / max(self.spike_window, 1),
            "e_actor": step_e_actor / max(self.spike_window, 1),
            "e_critic": step_e_critic / max(self.spike_window, 1),
        }

        if not commit:
            self._restore_snapshot(snapshot)
        return out
