import numpy as np
import control as cnt
import hummingbirdParam as P

class ctrlStateFeedbackIntegrator:
    def __init__(self):
        # Tuning parameters (all round same value, varied for unique poles)
        wn_th = 2.2
        zeta_th = 0.707
        wn_psi = 2.0
        zeta_psi = 2
        wn_phi = 2.0
        zeta_phi = 0.711

        # Longitudinal dynamics (these values are given in page 33 of manual)
        # State space equations are
        # x* =  A  * x + B
        # yr = C_r * x
        A_lon = np.array([[0, 1],
                          [0, 0]])
        B_lon = np.array([[0],
                          [P.b_theta]])
        C_lon = np.array([[1, 0]])  # output: theta

        # This is in Ethan's notes on INDEX 14
        A1_lon = np.block([
            [A_lon, np.zeros((2, 1))],
            [-C_lon, np.zeros((1, 1))]
        ])
        B1_lon = np.vstack((B_lon, np.zeros((1, 1))))

        # Get the poles (two for long dynamics, one for integrator) 
        p_lon = np.array(np.roots([
            1,
            2 * zeta_th * wn_th,
            wn_th ** 2
        ]).tolist() + [-wn_th / 2])

        print(A1_lon)
        print(B1_lon)
        print(p_lon)

        # We use the place to calculate the gains that would put the poles in specified locations
        K1_lon = cnt.place(A1_lon, B1_lon, p_lon)

        # Extract the gains
        self.k_th = K1_lon[0, 0]
        self.k_thdot = K1_lon[0, 1]
        self.ki_lon = K1_lon[0, 2]

        # Lateral dynamics (again, defined on page 34 of manual)
        a1 = P.ellT * P.m1 * P.g / (P.JT + P.J1z)
        A_lat = np.array([[0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [0, 0, 0, 0],
                          [a1, 0, 0, 0]])
        B_lat = np.array([[0],
                          [0],
                          [1 / P.J1x],
                          [0]])
        C_lat = np.array([[0, 1, 0, 0]])  # output: psi

        A1_lat = np.block([
            [A_lat, np.zeros((4, 1))],
            [-C_lat, np.zeros((1, 1))]
        ])
        B1_lat = np.vstack((B_lat, np.zeros((1, 1))))

        p_lat = np.concatenate([
            np.roots([1, 2 * zeta_phi * wn_phi, wn_phi ** 2]),
            np.roots([1, 2 * zeta_psi * wn_psi, wn_psi ** 2]),
            [-wn_psi / 2]
        ])

        print(A1_lat)
        print(B1_lat)
        print(p_lat)

        K1_lat = cnt.place(A1_lat, B1_lat, p_lat)
        self.k_phi = K1_lat[0, 0]
        self.k_psi = K1_lat[0, 1]
        self.k_phidot = K1_lat[0, 2]
        self.k_psidot = K1_lat[0, 3]
        self.ki_lat = K1_lat[0, 4]

        print('K_lon: [', self.k_th, ',', self.k_thdot, ']')
        print('ki_lon: ', self.ki_lon)
        print('K_lat: [', self.k_phi, ',', self.k_psi, ',', self.k_phidot, ',', self.k_psidot, ']')
        print('ki_lat: ', self.ki_lat)

        theta_max = 30.0 * np.pi / 180.0  # Max theta, rads
        self.Ts = P.Ts
        sigma = 0.05
        self.beta = (2 * sigma - self.Ts) / (2 * sigma + self.Ts)
        self.phi_d1 = 0.
        self.phi_dot = 0.
        self.theta_d1 = 0.
        self.theta_dot = 0.
        self.psi_d1 = 0.
        self.psi_dot = 0.
        self.integrator_th = 0.0
        self.error_th_d1 = 0.0
        self.integrator_psi = 0.0
        self.error_psi_d1 = 0.0

    def update(self, r: np.ndarray, y: np.ndarray):
        theta_ref = r[0][0]
        psi_ref = r[1][0]
        phi = y[0][0]
        theta = y[1][0]
        psi = y[2][0]

        force_equilibrium = P.m1 * P.g

        # Dirty derivatives
        self.phi_dot = self.beta * self.phi_dot + (1 - self.beta) * ((phi - self.phi_d1) / self.Ts)
        self.theta_dot = self.beta * self.theta_dot + (1 - self.beta) * ((theta - self.theta_d1) / self.Ts)
        self.psi_dot = self.beta * self.psi_dot + (1 - self.beta) * ((psi - self.psi_d1) / self.Ts)

        self.phi_d1 = phi
        self.theta_d1 = theta
        self.psi_d1 = psi

        # Integrator updates
        error_th = theta_ref - theta
        self.integrator_th += (self.Ts / 2.0) * (error_th + self.error_th_d1)
        self.error_th_d1 = error_th

        error_psi = psi_ref - psi
        self.integrator_psi += (self.Ts / 2.0) * (error_psi + self.error_psi_d1)
        self.error_psi_d1 = error_psi

        # Longitudinal control
        x_lon = np.array([[theta], [self.theta_dot]])
        force_unsat = force_equilibrium - self.k_th * x_lon[0, 0] - self.k_thdot * x_lon[1, 0] - self.ki_lon * self.integrator_th
        force = saturate(force_unsat, -P.force_max, P.force_max)

        # Lateral control
        x_lat = np.array([[phi], [psi], [self.phi_dot], [self.psi_dot]])
        torque_unsat = -self.k_phi * x_lat[0, 0] - self.k_psi * x_lat[1, 0] - self.k_phidot * x_lat[2, 0] - self.k_psidot * x_lat[3, 0] - self.ki_lat * self.integrator_psi
        torque = saturate(torque_unsat, -P.torque_max, P.torque_max)

        # Convert to PWM
        pwm = np.array([[force + torque / P.d],
                        [force - torque / P.d]]) / (2 * P.km)
        pwm = saturate(pwm, 0, 1)
        return pwm, np.array([[0], [theta_ref], [psi_ref]])


def saturate(u, low_limit, up_limit):
    if isinstance(u, float):
        u = min(max(u, low_limit), up_limit)
    else:
        for i in range(u.shape[0]):
            u[i][0] = min(max(u[i][0], low_limit), up_limit)
    return u
