import numpy as np


class OnlineActorCriticSNN:
    """
    Full spiking actor-critic network.

    Architecture
    -----------
    Input state
        -> rate-encoded input spikes over T ticks
        -> hidden LIF spiking layer
        -> actor LIF spiking output layer
        -> critic LIF spiking output neuron

    Action selection
    ----------------
    Actor output spike rates are converted to a softmax policy.

    Value estimate
    --------------
    Critic output spike rate is converted to a scalar value estimate.

    Learning
    --------
    Uses an approximate online actor-critic update with surrogate/rate-style
    gradients. This is not exact BPTT through spikes, but the network itself
    is fully spiking during forward execution.
    """

    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.1,
        hidden_size=32,
        spike_window=20,
        mem_decay=0.9,
        threshold=1.0,
        input_scale=1.0,
        value_scale=5.0,
        actor_lr=None,
        critic_lr=None,
        seed=None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.hidden_size = hidden_size

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

        self.rng = np.random.RandomState(seed)

        # Input -> hidden
        self.W_in = 0.2 * self.rng.randn(self.hidden_size, self.n_features)
        self.b_h = np.zeros(self.hidden_size)

        # Hidden -> actor output
        self.W_actor = 0.2 * self.rng.randn(self.n_actions, self.hidden_size)
        self.b_actor = np.zeros(self.n_actions)

        # Hidden -> critic output
        self.W_critic = 0.2 * self.rng.randn(self.hidden_size)
        self.b_critic = 0.0

    def _softmax(self, x):
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-12)

    def _normalize_state(self, state):
        """
        Simple normalization for rate encoding.
        If your observation is already in [0,1] or binary, this is fine.
        """
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        s = np.clip(s * self.input_scale, 0.0, 1.0)
        return s

    def _rate_encode(self, state):
        """
        Generate input spikes over time using Bernoulli rate coding.
        Returns shape (T, n_features)
        """
        s = self._normalize_state(state)
        spikes = (self.rng.rand(self.spike_window, self.n_features) < s[None, :]).astype(np.float64)
        return spikes

    def _lif_step(self, v, input_current):
        """
        One LIF-like step with soft reset after spike.
        """
        v = self.mem_decay * v + input_current
        spikes = (v >= self.threshold).astype(np.float64)
        v = v - spikes * self.threshold
        return v, spikes

    def _surrogate_grad(self, v):
        """
        Simple triangular surrogate derivative around threshold.
        """
        dist = np.abs(v - self.threshold)
        grad = np.maximum(0.0, 1.0 - dist / self.threshold)
        return grad

    def _forward(self, state):
        """
        Full spiking forward pass over spike_window ticks.

        Returns a dict containing:
        - input spike train
        - hidden spikes/rates
        - actor spikes/rates
        - critic spikes/rate
        - policy probs
        - value estimate
        - membrane traces for surrogate updates
        """
        x_spikes = self._rate_encode(state)  # (T, n_features)
        T = self.spike_window

        v_h = np.zeros(self.hidden_size)
        v_a = np.zeros(self.n_actions)
        v_c = 0.0

        h_spikes_hist = np.zeros((T, self.hidden_size))
        a_spikes_hist = np.zeros((T, self.n_actions))
        c_spikes_hist = np.zeros(T)

        v_h_hist = np.zeros((T, self.hidden_size))
        v_a_hist = np.zeros((T, self.n_actions))
        v_c_hist = np.zeros(T)

        for t in range(T):
            x_t = x_spikes[t]

            # Hidden spiking layer
            h_current = self.W_in @ x_t + self.b_h
            v_h, h_spikes = self._lif_step(v_h, h_current)

            # Actor spiking layer
            a_current = self.W_actor @ h_spikes + self.b_actor
            v_a, a_spikes = self._lif_step(v_a, a_current)

            # Critic spiking neuron
            c_current = np.dot(self.W_critic, h_spikes) + self.b_critic
            v_c_arr, c_spike_arr = self._lif_step(np.array([v_c]), np.array([c_current]))
            v_c = float(v_c_arr[0])
            c_spike = float(c_spike_arr[0])

            h_spikes_hist[t] = h_spikes
            a_spikes_hist[t] = a_spikes
            c_spikes_hist[t] = c_spike

            v_h_hist[t] = v_h
            v_a_hist[t] = v_a
            v_c_hist[t] = v_c

        x_rate = x_spikes.mean(axis=0)
        h_rate = h_spikes_hist.mean(axis=0)
        actor_rate = a_spikes_hist.mean(axis=0)
        critic_rate = float(c_spikes_hist.mean())

        # Use actor spike rates as logits
        probs = self._softmax(actor_rate)

        # Scale critic spike rate into a value estimate
        value = self.value_scale * critic_rate

        return {
            "x_spikes": x_spikes,
            "x_rate": x_rate,
            "h_spikes_hist": h_spikes_hist,
            "h_rate": h_rate,
            "a_spikes_hist": a_spikes_hist,
            "actor_rate": actor_rate,
            "c_spikes_hist": c_spikes_hist,
            "critic_rate": critic_rate,
            "probs": probs,
            "value": value,
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
        probs = out["probs"]

        # Match your current convention:
        # epsilon = probability of using learned policy
        if self.rng.rand() < self.epsilon:
            return self.rng.choice(self.n_actions, p=probs)

        return self.rng.randint(self.n_actions)

    def learn(self, state, action, reward, next_state, done=False):
        """
        Online actor-critic update.

        Forward pass is fully spiking.
        Weight update uses approximate surrogate/rate-based gradients.
        """
        cur = self._forward(state)
        nxt = self._forward(next_state)

        v_s = cur["value"]
        v_next = 0.0 if done else nxt["value"]

        td_error = reward + self.gamma * v_next - v_s

        x_rate = cur["x_rate"]
        h_rate = cur["h_rate"]
        actor_rate = cur["actor_rate"]
        probs = cur["probs"]

        # Average surrogate activity terms
        sg_h = self._surrogate_grad(cur["v_h_hist"]).mean(axis=0)   # (hidden,)
        sg_a = self._surrogate_grad(cur["v_a_hist"]).mean(axis=0)   # (actions,)
        sg_c = self._surrogate_grad(cur["v_c_hist"]).mean()         # scalar

        # --------------------
        # Critic output update
        # --------------------
        self.W_critic += self.critic_lr * td_error * sg_c * h_rate
        self.b_critic += self.critic_lr * td_error * sg_c

        # -------------------
        # Actor output update
        # -------------------
        # grad log pi wrt actor output rates/logits
        grad_log_pi = -probs
        grad_log_pi[action] += 1.0

        # hidden -> actor
        for a in range(self.n_actions):
            self.W_actor[a] += self.actor_lr * td_error * grad_log_pi[a] * sg_a[a] * h_rate
            self.b_actor[a] += self.actor_lr * td_error * grad_log_pi[a] * sg_a[a]

        # ---------------------------------------
        # Input -> hidden approximate backprop
        # ---------------------------------------
        # Push actor + critic learning signal back into hidden layer
        actor_back = np.dot((grad_log_pi * sg_a), self.W_actor)      # (hidden,)
        critic_back = td_error * sg_c * self.W_critic                # (hidden,)
        hidden_signal = td_error * actor_back + critic_back          # (hidden,)

        for h in range(self.hidden_size):
            self.W_in[h] += self.lr * hidden_signal[h] * sg_h[h] * x_rate
            self.b_h[h] += self.lr * hidden_signal[h] * sg_h[h]

        return td_error