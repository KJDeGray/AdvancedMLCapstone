import numpy as np


class RecurrentActorCritic:
    """
    Lightweight recurrent ANN actor-critic.

    The recurrent layer is a tanh reservoir/state encoder updated online:

        h_t = tanh(W_in x_t + W_rec h_{t-1} + b)

    The actor and critic are trained with TD(0)-style actor-critic updates.
    To keep the baseline simple and stable, the recurrent/input weights are
    fixed after initialization and only the actor/critic readouts are learned.
    """

    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.1,
        hidden_size=48,
        recurrent_scale=0.8,
        input_scale=1.0,
        actor_lr=None,
        critic_lr=None,
        seed=None,
    ):
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.hidden_size = int(hidden_size)

        self.gamma = float(reward_decay)
        self.epsilon = float(e_greedy)
        self.input_scale = float(input_scale)

        self.actor_lr = learning_rate if actor_lr is None else actor_lr
        self.critic_lr = learning_rate if critic_lr is None else critic_lr

        self.rng = np.random.RandomState(seed)

        self.W_in = self.rng.randn(self.hidden_size, self.n_features) / np.sqrt(max(self.n_features, 1))
        self.W_rec = self.rng.randn(self.hidden_size, self.hidden_size) / np.sqrt(max(self.hidden_size, 1))

        eigvals = np.linalg.eigvals(self.W_rec)
        radius = np.max(np.abs(eigvals))
        if radius > 1e-12:
            self.W_rec *= recurrent_scale / radius

        self.b_h = np.zeros(self.hidden_size)

        # Readout input is [current observation, recurrent state, bias].
        self.readout_size = self.n_features + self.hidden_size + 1
        self.actor_w = 0.01 * self.rng.randn(self.n_actions, self.readout_size)
        self.critic_w = 0.01 * self.rng.randn(self.readout_size)

        self.reset_state()

    def reset_state(self):
        self.h = np.zeros(self.hidden_size)
        self.last_features = None
        self.last_action = None

    def _normalize_state(self, state):
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        if len(s) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(s)}")
        return np.clip(s * self.input_scale, 0.0, 1.0)

    def _softmax(self, logits):
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / (np.sum(exp_logits) + 1e-12)

    def _features_from_state(self, state, commit=True):
        x = self._normalize_state(state)
        h_next = np.tanh(self.W_in @ x + self.W_rec @ self.h + self.b_h)
        features = np.concatenate([x, h_next, [1.0]])
        if commit:
            self.h = h_next
        return features

    def _policy_from_features(self, features):
        return self._softmax(self.actor_w @ features)

    def value_from_features(self, features):
        return float(np.dot(self.critic_w, features))

    def policy(self, state):
        features = self._features_from_state(state, commit=False)
        return self._policy_from_features(features)

    def value(self, state):
        features = self._features_from_state(state, commit=False)
        return self.value_from_features(features)

    def choose_action(self, state):
        features = self._features_from_state(state, commit=True)
        probs = self._policy_from_features(features)

        if self.rng.rand() < self.epsilon:
            action = int(self.rng.choice(self.n_actions, p=probs))
        else:
            action = int(self.rng.randint(self.n_actions))

        self.last_features = features
        self.last_action = action
        return action

    def learn(self, state, action, reward, next_state, done=False):
        features = self.last_features
        if features is None or self.last_action != int(action):
            features = self._features_from_state(state, commit=True)

        next_features = self._features_from_state(next_state, commit=False)
        v_s = self.value_from_features(features)
        v_next = 0.0 if done else self.value_from_features(next_features)
        td_error = float(reward + self.gamma * v_next - v_s)

        self.critic_w += self.critic_lr * td_error * features

        probs = self._policy_from_features(features)
        for action_idx in range(self.n_actions):
            grad_log_pi = ((1.0 if action_idx == int(action) else 0.0) - probs[action_idx]) * features
            self.actor_w[action_idx] += self.actor_lr * td_error * grad_log_pi

        self.last_features = None
        self.last_action = None
        return td_error
