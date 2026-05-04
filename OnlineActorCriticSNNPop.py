import numpy as np


class OnlineActorCriticSNNPop:
    """
    Full spiking actor-critic with population-coded outputs.

    Architecture
    ------------
    state
      -> rate-encoded input spikes
      -> hidden LIF spiking layer
      -> actor output populations (multiple spiking neurons per action)
      -> critic population (multiple spiking neurons for value)

    Output decoding
    ---------------
    - Actor: each action has a population of neurons. The action logit is the
      mean spike rate of that population.
    - Critic: value is decoded from the mean spike rate of the critic population.

    Learning
    --------
    Approximate online actor-critic update using surrogate/rate-style gradients.
    Forward pass is fully spiking.
    """

    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.001,
        reward_decay=0.9,
        e_greedy=0.1,
        hidden_size=32,
        actor_pop_size=4,
        critic_pop_size=8,
        spike_window=50,
        mem_decay=0.9,
        threshold=0.5,
        input_scale=1.0,
        value_scale=5.0,
        actor_output_gain=5.0,
        actor_lr=None,
        critic_lr=None,
        seed=None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.actor_pop_size = actor_pop_size
        self.critic_pop_size = critic_pop_size

        self.n_actor_neurons = n_actions * actor_pop_size

        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.lr = learning_rate
        self.actor_lr = learning_rate if actor_lr is None else actor_lr
        self.critic_lr = learning_rate if critic_lr is None else critic_lr

        self.spike_window = int(spike_window)
        self.mem_decay = float(mem_decay)
        self.threshold = float(threshold)
        self.input_scale = float(input_scale)
        self.value_scale = float(value_scale)
        self.actor_output_gain = float(actor_output_gain)

        self.rng = np.random.RandomState(seed)
        self._last_spike_count = 0.0
        self._last_forward = None
        self._last_action = None

        # Input -> hidden
        self.W_in = 0.1 * self.rng.randn(self.hidden_size, self.n_features)
        self.b_h = np.zeros(self.hidden_size)

        # Hidden -> actor population
        self.W_actor = 0.1 * self.rng.randn(self.n_actor_neurons, self.hidden_size)
        self.b_actor = np.zeros(self.n_actor_neurons)

        # Hidden -> critic population
        self.W_critic = 0.1 * self.rng.randn(self.critic_pop_size, self.hidden_size)
        self.b_critic = np.zeros(self.critic_pop_size)

    def _softmax(self, x):
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-12)

    def _normalize_state(self, state):
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        s = np.clip(s * self.input_scale, 0.0, 1.0)
        return s

    def _rate_encode(self, state):
        """
        Bernoulli rate coding over spike_window ticks.
        Output shape: (T, n_features)
        """
        s = self._normalize_state(state)
        spikes = (self.rng.rand(self.spike_window, self.n_features) < s[None, :]).astype(np.float64)
        return spikes

    def _lif_step(self, v, input_current):
        v = self.mem_decay * v + input_current
        spikes = (v >= self.threshold).astype(np.float64)
        v = v - spikes * self.threshold
        return v, spikes

    def _surrogate_grad(self, v):
        """
        Simple triangular surrogate gradient around threshold.
        """
        dist = np.abs(v - self.threshold)
        grad = np.maximum(0.0, 1.0 - dist / max(self.threshold, 1e-6))
        return grad

    def _decode_actor_populations(self, actor_rate_neurons):
        """
        actor_rate_neurons shape: (n_actions * actor_pop_size,)
        Returns per-action logits/rates shape: (n_actions,)
        """
        actor_rate_matrix = actor_rate_neurons.reshape(self.n_actions, self.actor_pop_size)
        action_rates = actor_rate_matrix.mean(axis=1)
        return action_rates

    def _forward(self, state):
        x_spikes = self._rate_encode(state)
        T = self.spike_window

        v_h = np.zeros(self.hidden_size)
        v_a = np.zeros(self.n_actor_neurons)
        v_c = np.zeros(self.critic_pop_size)

        h_spikes_hist = np.zeros((T, self.hidden_size))
        a_spikes_hist = np.zeros((T, self.n_actor_neurons))
        c_spikes_hist = np.zeros((T, self.critic_pop_size))

        v_h_hist = np.zeros((T, self.hidden_size))
        v_a_hist = np.zeros((T, self.n_actor_neurons))
        v_c_hist = np.zeros((T, self.critic_pop_size))

        for t in range(T):
            x_t = x_spikes[t]

            # Hidden layer
            h_current = self.W_in @ x_t + self.b_h
            v_h, h_spikes = self._lif_step(v_h, h_current)

            # Actor population
            a_current = self.W_actor @ h_spikes + self.b_actor
            v_a, a_spikes = self._lif_step(v_a, a_current)

            # Critic population
            c_current = self.W_critic @ h_spikes + self.b_critic
            v_c, c_spikes = self._lif_step(v_c, c_current)

            h_spikes_hist[t] = h_spikes
            a_spikes_hist[t] = a_spikes
            c_spikes_hist[t] = c_spikes

            v_h_hist[t] = v_h
            v_a_hist[t] = v_a
            v_c_hist[t] = v_c

        x_rate = x_spikes.mean(axis=0)
        h_rate = h_spikes_hist.mean(axis=0)

        actor_rate_neurons = a_spikes_hist.mean(axis=0)
        critic_rate_neurons = c_spikes_hist.mean(axis=0)

        action_rates = self._decode_actor_populations(actor_rate_neurons)
        probs = self._softmax(self.actor_output_gain * action_rates)

        critic_rate = float(np.mean(critic_rate_neurons))
        value = self.value_scale * critic_rate
        spike_count = float(
            np.sum(x_spikes)
            + np.sum(h_spikes_hist)
            + np.sum(a_spikes_hist)
            + np.sum(c_spikes_hist)
        )

        return {
            "x_spikes": x_spikes,
            "x_rate": x_rate,
            "h_spikes_hist": h_spikes_hist,
            "h_rate": h_rate,
            "a_spikes_hist": a_spikes_hist,
            "actor_rate_neurons": actor_rate_neurons,
            "action_rates": action_rates,
            "c_spikes_hist": c_spikes_hist,
            "critic_rate_neurons": critic_rate_neurons,
            "critic_rate": critic_rate,
            "probs": probs,
            "value": value,
            "spike_count": spike_count,
            "v_h_hist": v_h_hist,
            "v_a_hist": v_a_hist,
            "v_c_hist": v_c_hist,
        }

    def policy(self, state):
        out = self._forward(state)
        return out["probs"]

    def value(self, state):
        out = self._forward(state)
        return out["value"]

    def choose_action(self, state):
        out = self._forward(state)
        self._last_forward = out
        self._last_spike_count = out["spike_count"]
        probs = out["probs"]

        # same convention as your other actor-critic:
        # epsilon = probability of using learned policy
        if self.rng.rand() < self.epsilon:
            action = int(self.rng.choice(self.n_actions, p=probs))
        else:
            action = int(self.rng.randint(self.n_actions))

        self._last_action = action
        return action

    def last_spike_count(self):
        return float(self._last_spike_count)

    def learn(self, state, action, reward, next_state, done=False):
        if self._last_forward is not None and self._last_action == int(action):
            cur = self._last_forward
        else:
            cur = self._forward(state)
        nxt = self._forward(next_state)

        v_s = cur["value"]
        v_next = 0.0 if done else nxt["value"]

        td_error = reward + self.gamma * v_next - v_s

        x_rate = cur["x_rate"]
        h_rate = cur["h_rate"]
        probs = cur["probs"]

        sg_h = self._surrogate_grad(cur["v_h_hist"]).mean(axis=0)   # (hidden,)
        sg_a = self._surrogate_grad(cur["v_a_hist"]).mean(axis=0)   # (n_actor_neurons,)
        sg_c = self._surrogate_grad(cur["v_c_hist"]).mean(axis=0)   # (critic_pop_size,)

        # -----------------------
        # Critic population update
        # -----------------------
        for c in range(self.critic_pop_size):
            self.W_critic[c] += self.critic_lr * td_error * sg_c[c] * h_rate
            self.b_critic[c] += self.critic_lr * td_error * sg_c[c]

        # ----------------------
        # Actor population update
        # ----------------------
        # policy gradient on action-level decoded outputs
        grad_log_pi_actions = -probs
        grad_log_pi_actions[action] += 1.0

        # apply same action-level signal to each neuron in that action's population
        for a in range(self.n_actions):
            start = a * self.actor_pop_size
            end = (a + 1) * self.actor_pop_size
            for j in range(start, end):
                self.W_actor[j] += self.actor_lr * td_error * grad_log_pi_actions[a] * sg_a[j] * h_rate
                self.b_actor[j] += self.actor_lr * td_error * grad_log_pi_actions[a] * sg_a[j]

        # ----------------------------------------
        # Approximate hidden-layer credit assignment
        # ----------------------------------------
        actor_back = np.zeros(self.hidden_size)
        for a in range(self.n_actions):
            start = a * self.actor_pop_size
            end = (a + 1) * self.actor_pop_size
            pop_signal = grad_log_pi_actions[a]
            for j in range(start, end):
                actor_back += pop_signal * sg_a[j] * self.W_actor[j]

        critic_back = np.zeros(self.hidden_size)
        for c in range(self.critic_pop_size):
            critic_back += td_error * sg_c[c] * self.W_critic[c]

        hidden_signal = td_error * actor_back + critic_back

        for h in range(self.hidden_size):
            self.W_in[h] += self.lr * hidden_signal[h] * sg_h[h] * x_rate
            self.b_h[h] += self.lr * hidden_signal[h] * sg_h[h]

        self._last_forward = None
        self._last_action = None
        return td_error
