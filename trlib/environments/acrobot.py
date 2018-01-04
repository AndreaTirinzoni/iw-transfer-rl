import gym
import numpy as np
from gym import spaces
from scipy.integrate import odeint

"""
The Acrobot environment

Info
----
  - State space: 4D Box (theta1,theta2,theta_dot1,theta_dot2)
  - Action space: Discrete (-5 or 5)
  - Parameters: lengths l1 and l2, masses m1 and m2, centers of mass mu1 and m2

References
----------
  - Ernst, Damien, Pierre Geurts, and Louis Wehenkel
    Tree-based batch mode reinforcement learning
    Journal of Machine Learning Research 6.Apr (2005): 503-556
"""


class Acrobot(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, m1 = 1.0, m2 = 1.0, l1 = 1.0, l2 = 1.0, mu1 = 0.01, mu2 = 0.01):
        self.horizon = 100
        self.gamma = .95
        self.state_dim = 4
        self.action_dim = 1
        
        self._g = 9.81
        self._M1 = m1
        self._M2 = m2
        self._L1 = l1
        self._L2 = l2
        self._mu1 = mu1
        self._mu2 = mu2
        self._dt = .1
        self.torques = [-5.0, 5.0]

        # gym attributes
        self.viewer = None
        high = np.array([np.inf, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = spaces.Discrete(2)

        # initialize state
        self.seed()
        self.reset()

    def step(self, a):
        u = self.torques[int(a)]
        sa = np.append(self._state, u)
        new_state = odeint(self._dpds,
                           sa,
                           [0, self._dt],
                           rtol=1e-5, atol=1e-5, mxstep=2000)

        x = new_state[-1, :-1]

        k = round((x[0] - np.pi) / (2 * np.pi))
        o = np.array([2 * k * np.pi + np.pi, 0., 0., 0.])
        d = np.linalg.norm(x - o)

        x[0] = self._wrap2pi(x[0])
        x[1] = self._wrap2pi(x[1])

        self._state = x

        reward = 0.0
        if d < 1:
            self._absorbing = True
            reward = 1 - d

        return self.get_state(), reward, self._absorbing, {}

    def reset(self, state=None):
        self._absorbing = False
        if state is None:
            theta1 = self._wrap2pi(np.random.uniform(low=-np.pi + 1,
                                                          high=np.pi - 1))
            theta2 = dTheta1 = dTheta2 = 0
        else:
            theta1 = self._wrap2pi(state[0])
            theta2 = self._wrap2pi(state[1])
            dTheta1 = state[2]
            dTheta2 = state[3]

        self._state = np.array([theta1, theta2, dTheta1, dTheta2])

        return self.get_state()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering

        s = self._state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

        p1 = [-self._L1 *
              np.cos(s[0]), self._L1 * np.sin(s[0])]

        p2 = [p1[0] - self._L2 * np.cos(s[0] + s[1]),
              p1[1] + self._L2 * np.sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th) in zip(xys, thetas):
            l, r, t, b = 0, 1, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_state(self):
        return self._state

    def _dpds(self, state_action, t):
        theta1 = state_action[0]
        theta2 = state_action[1]
        d_theta1 = state_action[2]
        d_theta2 = state_action[3]
        u = state_action[-1]

        d11 = self._M1 * self._L1 * self._L1 + self._M2 * \
            (self._L1 * self._L1 + self._L2 * self._L2 + 2 * self._L1 *
             self._L2 * np.cos(theta2))
        d22 = self._M2 * self._L2 * self._L2
        d12 = self._M2 * (self._L2 * self._L2 + self._L1 * self._L2 *
                          np.cos(theta2))
        c1 = -self._M2 * self._L1 * self._L2 * d_theta2 * \
            (2 * d_theta1 + d_theta2 * np.sin(theta2))
        c2 = self._M2 * self._L1 * self._L2 * d_theta1 * d_theta1 * \
            np.sin(theta2)
        phi1 = (self._M1 * self._L1 + self._M2 * self._L1) * self._g * \
            np.sin(theta1) + self._M2 * self._L2 * self._g * \
            np.sin(theta1 + theta2)
        phi2 = self._M2 * self._L2 * self._g * np.sin(theta1 + theta2)

        diff_theta1 = d_theta1
        diff_theta2 = d_theta2
        d12d22 = d12 / d22
        diff_diff_theta1 = (-self._mu1 * d_theta1 - d12d22 * u + d12d22 *
                            self._mu2 * d_theta2 + d12d22 * c2 + d12d22 *
                            phi2 - c1 - phi1) / (d11 - (d12d22 * d12))
        diff_diff_theta2 = (u - self._mu2 * d_theta2 - d12 * diff_diff_theta1 -
                            c2 - phi2) / d22

        return diff_theta1, diff_theta2, diff_diff_theta1, diff_diff_theta2, 0.

    def _wrap2pi(self, value):
        tmp = value - -np.pi
        width = 2 * np.pi
        tmp -= width * np.floor(tmp / width)

        return tmp + -np.pi