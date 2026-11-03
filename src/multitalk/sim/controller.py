"""Differential IK controller, adapted from FR3Env (MIT, (c) 2022 Bolun Dai)."""

from __future__ import annotations

import numpy as np
import proxsuite

Q_NOMINAL = np.array(
    [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397, 0.0, 0.0]
)


class DiffIK:
    def __init__(self) -> None:
        self.epsilon = 0.01
        self.kp_joint_centering = 0.01
        self.n = 9
        self.qp = proxsuite.proxqp.dense.QP(self.n, 0, 0)
        self.qp_initialized = False

    def __call__(self, v_des: np.ndarray, q: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        H, g = self._qp_params(v_des, q, jacobian)
        if not self.qp_initialized:
            self.qp.init(H=H, g=g)
            self.qp.settings.eps_abs = 1.0e-6
            self.qp.settings.max_iter = 20
            self.qp_initialized = True
        else:
            self.qp.update(H=H, g=g)
        self.qp.solve()
        return self.qp.results.x

    def _qp_params(self, v_des, q, jacobian):
        nullspace = np.eye(self.n) - np.linalg.pinv(jacobian) @ jacobian
        nullspace_quadratic = nullspace.T @ nullspace
        H = 2 * (jacobian.T @ jacobian + self.epsilon * nullspace_quadratic)
        dq_nominal = self.kp_joint_centering * (Q_NOMINAL - q)[:, np.newaxis]
        g = -2 * (v_des.T @ jacobian + self.epsilon * dq_nominal.T @ nullspace_quadratic).T
        return H, g.ravel()
