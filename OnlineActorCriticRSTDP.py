import numpy as np


class OnlineActorCriticRSTDP:
    """
    Persistent-state spiking actor-critic with three-factor updates.

    The network uses rate-coded input spikes, LIF hidden/output neurons,
    eligibility traces, and TD-error modulation. It is intentionally small
    and NumPy-only so it can be compared directly with the other project
    baselines.
    """

    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.002,
        reward_decay=0.9,
        e_greedy=0.1,
        hidden_size=48,
        actor_pop_size=6,
        critic_pop_size=8,
        spike_window=30,
        mem_decay=0.9,
        trace_decay=0.85,
        eligibility_decay=0.95,
        threshold=3.0,
        input_scale=1.0,
        value_scale=6.0,
        actor_output_gain=8.0,
        weight_clip=2.0,
        target_spike_count=850.0,
        homeostatic_lr=0.0015,
        weight_decay=0.0001,
        actor_bias_modulation=0.25,
        actor_lr=None,
        critic_lr=None,
        input_lr=None,
        seed=None,
    ):
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.hidden_size = int(hidden_size)
        self.actor_pop_size = int(actor_pop_size)
        self.critic_pop_size = int(critic_pop_size)
        self.n_actor_neurons = self.n_actions * self.actor_pop_size

        self.gamma = float(reward_decay)
        self.epsilon = float(e_greedy)

        self.actor_lr = learning_rate if actor_lr is None else actor_lr
        self.critic_lr = learning_rate if critic_lr is None else critic_lr
        self.input_lr = learning_rate * 0.25 if input_lr is None else input_lr

        self.spike_window = int(spike_window)
        self.mem_decay = float(mem_decay)
        self.trace_decay = float(trace_decay)
        self.eligibility_decay = float(eligibility_decay)
        self.threshold = float(threshold)
        self.input_scale = float(input_scale)
        self.value_scale = float(value_scale)
        self.actor_output_gain = float(actor_output_gain)
        self.weight_clip = float(weight_clip)
        self.target_spike_count = float(target_spike_count)
        self.homeostatic_lr = float(homeostatic_lr)
        self.weight_decay = float(weight_decay)
        self.actor_bias_modulation = float(actor_bias_modulation)

        self.rng = np.random.RandomState(seed)
        self.learning_frozen = False

        # Positive-biased input weights make rate-coded binary observations
        # produce useful hidden activity early in training.
        self.W_in = self.rng.uniform(0.0, 0.08, size=(self.hidden_size, self.n_features))
        self.b_h = self.rng.uniform(0.0, 0.01, size=self.hidden_size)

        self.W_actor = self.rng.uniform(0.0, 0.05, size=(self.n_actor_neurons, self.hidden_size))
        self.b_actor = self.rng.uniform(0.0, 0.005, size=self.n_actor_neurons)

        self.W_critic = self.rng.uniform(0.0, 0.05, size=(self.critic_pop_size, self.hidden_size))
        self.b_critic = self.rng.uniform(0.0, 0.005, size=self.critic_pop_size)

        self.reset_state()

    def freeze_learning(self):
        self.learning_frozen = True

    def unfreeze_learning(self):
        self.learning_frozen = False

    def reset_state(self):
        self.v_h = np.zeros(self.hidden_size)
        self.v_actor = np.zeros(self.n_actor_neurons)
        self.v_critic = np.zeros(self.critic_pop_size)

        self.x_trace = np.zeros(self.n_features)
        self.h_trace = np.zeros(self.hidden_size)
        self.actor_trace = np.zeros(self.n_actor_neurons)
        self.critic_trace = np.zeros(self.critic_pop_size)

        self.e_in = np.zeros((self.hidden_size, self.n_features))
        self.e_actor = np.zeros((self.n_actor_neurons, self.hidden_size))
        self.e_critic = np.zeros((self.critic_pop_size, self.hidden_size))

        self.last_forward = None
        self.last_action = None

    def _softmax(self, x):
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-12)

    def _normalize_state(self, state):
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        if len(s) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(s)}")
        return np.clip(s * self.input_scale, 0.0, 1.0)

    def _rate_encode(self, state):
        s = self._normalize_state(state)
        return (self.rng.rand(self.spike_window, self.n_features) < s[None, :]).astype(np.float64)

    def _lif_step(self, v, current):
        v = self.mem_decay * v + current
        spikes = (v >= self.threshold).astype(np.float64)
        v = v - spikes * self.threshold
        return v, spikes

    def _decode_actor(self, actor_spike_counts):
        rates = actor_spike_counts.reshape(self.n_actions, self.actor_pop_size).mean(axis=1)
        return rates

    def _state_snapshot(self):
        return (
            self.v_h.copy(),
            self.v_actor.copy(),
            self.v_critic.copy(),
            self.x_trace.copy(),
            self.h_trace.copy(),
            self.actor_trace.copy(),
            self.critic_trace.copy(),
            self.e_in.copy(),
            self.e_actor.copy(),
            self.e_critic.copy(),
        )

    def _restore_snapshot(self, snapshot):
        (
            self.v_h,
            self.v_actor,
            self.v_critic,
            self.x_trace,
            self.h_trace,
            self.actor_trace,
            self.critic_trace,
            self.e_in,
            self.e_actor,
            self.e_critic,
        ) = [item.copy() for item in snapshot]

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
            h_current = self.W_in @ x_t + self.b_h
            self.v_h, h_spikes = self._lif_step(self.v_h, h_current)

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

    def policy(self, state):
        return self._forward(state, commit=False)["probs"]

    def value(self, state):
        return self._forward(state, commit=False)["value"]

    def choose_action(self, state):
        out = self._forward(state, commit=True)
        probs = out["probs"]

        if self.rng.rand() < self.epsilon:
            action = int(self.rng.choice(self.n_actions, p=probs))
        else:
            action = int(self.rng.randint(self.n_actions))

        self.last_forward = out
        self.last_action = action
        return action

    def learn(self, state, action, reward, next_state, done=False):
        cur = self.last_forward
        if cur is None or self.last_action != action:
            cur = self._forward(state, commit=True)

        next_value = 0.0 if done else self._forward(next_state, commit=False)["value"]
        td_error = float(reward + self.gamma * next_value - cur["value"])
        if self.learning_frozen:
            self.last_forward = None
            self.last_action = None
            return td_error

        probs = cur["probs"]
        grad_log_pi = -probs
        grad_log_pi[int(action)] += 1.0

        actor_mod = np.repeat(grad_log_pi, self.actor_pop_size)
        d_actor = self.actor_lr * td_error * actor_mod[:, None] * cur["e_actor"]
        d_critic = self.critic_lr * td_error * cur["e_critic"]

        # Hidden synapses receive a local three-factor signal: eligibility
        # times TD-modulated downstream actor/critic contribution.
        actor_back = actor_mod @ self.W_actor
        critic_back = np.mean(self.W_critic, axis=0)
        hidden_mod = td_error * (0.5 * actor_back + critic_back)
        d_in = self.input_lr * hidden_mod[:, None] * cur["e_in"]

        self.W_actor = (1.0 - self.weight_decay) * self.W_actor + d_actor
        self.W_critic = (1.0 - self.weight_decay) * self.W_critic + d_critic
        self.W_in = (1.0 - self.weight_decay) * self.W_in + d_in

        self.W_actor = np.clip(self.W_actor, 0.0, self.weight_clip)
        self.W_critic = np.clip(self.W_critic, 0.0, self.weight_clip)
        self.W_in = np.clip(self.W_in, 0.0, self.weight_clip)

        # Biases are modulatory terms, not STDP traces, but help avoid silent
        # output populations during early training.
        action_mask = np.repeat(grad_log_pi, self.actor_pop_size)
        self.b_actor += self.actor_bias_modulation * self.actor_lr * td_error * action_mask
        self.b_critic += 0.05 * self.critic_lr * td_error
        self.b_h += 0.02 * self.input_lr * hidden_mod

        spike_error = (cur["spike_count"] - self.target_spike_count) / max(self.target_spike_count, 1.0)
        self.b_h -= self.homeostatic_lr * spike_error
        self.b_actor -= self.homeostatic_lr * spike_error
        self.b_critic -= self.homeostatic_lr * spike_error

        self.b_actor = np.clip(self.b_actor, -1.0, 1.0)
        self.b_critic = np.clip(self.b_critic, -1.0, 1.0)
        self.b_h = np.clip(self.b_h, -1.0, 1.0)

        self.last_forward = None
        self.last_action = None
        return td_error

    def last_spike_count(self):
        if self.last_forward is None:
            return 0.0
        return float(self.last_forward["spike_count"])
