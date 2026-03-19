import numpy as np


class OnlineActorCritic:
    """
    Simple online actor-critic with linear function approximation.

    - Actor: softmax policy over actions
    - Critic: linear state-value function V(s)
    - Online TD(0) update every step

    This is intentionally lightweight so it matches the style of the
    rest of your project: one model per SU, choose_action(...) each step,
    then learn(...) immediately after observing reward and next state.
    """

    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.1,
        actor_lr=None,
        critic_lr=None,
        seed=None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.actor_lr = learning_rate if actor_lr is None else actor_lr
        self.critic_lr = learning_rate if critic_lr is None else critic_lr

        self.rng = np.random.RandomState(seed)

        # Add one extra feature for bias
        self.actor_w = 0.01 * self.rng.randn(self.n_actions, self.n_features + 1)
        self.critic_w = 0.01 * self.rng.randn(self.n_features + 1)

    def _augment_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        return np.append(state, 1.0)  # bias term

    def _softmax(self, logits):
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / (np.sum(exp_logits) + 1e-12)

    def _policy(self, state):
        x = self._augment_state(state)
        logits = self.actor_w @ x
        probs = self._softmax(logits)
        return probs

    def value(self, state):
        x = self._augment_state(state)
        return float(np.dot(self.critic_w, x))

    def choose_action(self, state):
        probs = self._policy(state)

        if self.rng.rand() < self.epsilon:
            return self.rng.choice(self.n_actions, p=probs)   # exploit policy
        return self.rng.randint(self.n_actions)               # random explore

    def learn(self, state, action, reward, next_state, done=False):
        """
        Online actor-critic TD(0) update.

        delta = r + gamma * V(s') - V(s)

        Critic:
            w_v <- w_v + alpha_v * delta * x

        Actor:
            For softmax policy with preference h_a = w_a^T x,
            grad log pi(a|s) = x * (1_{a=j} - pi_j)

            w_actor[j] <- w_actor[j] + alpha_a * delta * grad_j
        """
        x = self._augment_state(state)
        x_next = self._augment_state(next_state)

        v_s = float(np.dot(self.critic_w, x))
        v_next = 0.0 if done else float(np.dot(self.critic_w, x_next))

        td_error = reward + self.gamma * v_next - v_s

        # Critic update
        self.critic_w += self.critic_lr * td_error * x

        # Actor update
        probs = self._softmax(self.actor_w @ x)
        for j in range(self.n_actions):
            grad_log_pi = ((1.0 if j == action else 0.0) - probs[j]) * x
            self.actor_w[j] += self.actor_lr * td_error * grad_log_pi

        return td_error