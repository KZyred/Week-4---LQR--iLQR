import numpy as np
import quadrotor

class solver4:
    def __init__(self, horizon_length = 1000):
        self.m = quadrotor.MASS
        self.r = quadrotor.LENGTH
        self.I = quadrotor.INERTIA
        self.g = quadrotor.GRAVITY
        self.dt = quadrotor.DELTA_T
        self.N = horizon_length
        # Part 1 Qs and Rs
        self.Q1 = np.array([[1e+06, 0., 0., 0., 0., 0.],
                           [0., 1e+03, 0., 0., 0., 0.],
                           [0., 0., 1e+06, 0., 0., 0.],
                           [0., 0., 0., 1e+03, 0., 0.],
                           [0., 0., 0., 0., 1e+06, 0.],
                           [0., 0., 0., 0., 0., 1e+03]])
        self.Q2 = np.array([[1e+01, 0., 0., 0., 0., 0.],
                            [0., 1e+00, 0., 0., 0., 0.],
                            [0., 0., 1e+01, 0., 0., 0.],
                            [0., 0., 0., 1e+00, 0., 0.],
                            [0., 0., 0., 0., 2e+02, 0.],
                            [0., 0., 0., 0., 0., 2e+00]])
        self.R1 = np.array([[1e-02, 0.],
                           [0., 1e-02]])
        self.R2 = np.array([[1e+01, 0.],
                            [0., 1e+01]])
        # 1. Inital Guesses
        # khởi tạo u*0, u*1,... ,u*N-1
        self.ustar = np.tile(np.array([[self.m * self.g / 2, self.m * self.g / 2]]).transpose(), (1, self.N))
        # khởi tạo x*0, x*1,... ,x*N-1
        self.xstar = self.get_states(np.array([0., 0., 0., 0., 0., 0.]), self.ustar)
        
        # 5 line Search
        self.prevCost = 0
        self.alpha = 1

    def get_states(self, x0, u):
        xstar = np.tile(np.array([[0., 0., 0., 0., 0., 0.]]).transpose(), (1, self.N + 1))
        xstar[:, 0] = x0
        for i, control in enumerate(u.transpose()):
            xstar[:, i + 1] = quadrotor.get_next_state(xstar[:, i], control)  # trạng thái mong muốn tiếp theo?
        return xstar

    # returns the cost of a trajectory z with control trajectory u (using the cost function you wrote in question 1)
    def compute_cost(self, z, u):
        J = [0] * (u.shape[1]) 
        # print(u.shape[1]) = 1000
        for i, (state, control) in enumerate(zip(z.transpose(), u.transpose())):
            if i == 500:  # lý do 500: bằng horizon_length = 1000/2 (đảm bảo ở giữa thì nhảy vào vị trí này)
                J[i] = (((state - np.array([3., 0., 3., 0., np.pi / 2, 0.])).reshape((6, 1)).transpose()) @ self.Q1 @ (state - np.array([3., 0., 3., 0., np.pi / 2, 0.])).reshape((6, 1))) + ((control - self.ustar[:, i]).reshape((2, 1)).transpose() @ self.R1 @ (control - self.ustar[:, i]).reshape((2, 1)))
            else:
                J[i] = (((state - np.array([0., 0., 0., 0., 0., 0.])).reshape((6, 1)).transpose()) @ self.Q2 @ (state - np.array([0., 0., 0., 0., 0., 0.])).reshape((6, 1))) + ((control - self.ustar[:, i]).reshape((2, 1)).transpose() @ self.R2 @ (control - self.ustar[:, i]).reshape((2, 1)))
        return max(J).item()

    # returns the quadratic approximation (Hessian matrices and Jacobians) of the cost function when approximated along the trajectory z with control trajectory u.
    # xấp xỉ Taylor bậc 2 (hàm chi phí)
    def get_quadratic_approximation_cost(self, z, u):
        An = [0] * (u.shape[1])
        Bn = [0] * (u.shape[1])
        Qn = [0] * (u.shape[1])
        qn = [0] * (u.shape[1])
        Rn = [0] * (u.shape[1])
        rn = [0] * (u.shape[1])
        
        for i, (state, control) in enumerate(zip(z.transpose(), u.transpose())):
            An[i], Bn[i] = self.get_linearization(state, control)
            if i == 500:
                Qn[i] = self.Q1
                qn[i] = (self.Q1 @ (state - np.array([3., 0., 3., 0., np.pi / 2, 0.])).reshape((6, 1)))
                Rn[i] = self.R1
                rn[i] = (self.R1 @ (control - self.ustar[:, i]).reshape((2, 1)))
            else:
                Qn[i] = self.Q2
                qn[i] = (self.Q2 @ (state - np.array([0., 0., 0., 0., 0., 0.])).reshape((6, 1)))
                Rn[i] = self.R2
                rn[i] = (self.R2 @ (control - self.ustar[:, i]).reshape((2, 1)))
        return An, Bn, Qn, qn, Rn, rn

    # giống bên LQR, thay đổi theo u và z
    # xấp xỉ Taylor bậc 1 (hệ động lực)
    def get_linearization(self, z, u):
        A = np.array([[1., self.dt, 0., 0., 0., 0.],
                      [0., 1., 0., 0., (-(u[0] + u[1]) * self.dt * np.cos(z[4])) / self.m, 0.],
                      [0., 0., 1., self.dt, 0., 0.],
                      [0., 0., 0., 1., (-(u[0] + u[1]) * self.dt * np.sin(z[4])) / self.m, 0.],
                      [0., 0., 0., 0., 1., self.dt],
                      [0., 0., 0., 0., 0., 1.]])
        B = np.array([[0., 0.],
                      [-(self.dt * np.sin(z[4])) / self.m, -(self.dt * np.sin(z[4])) / self.m],
                      [0., 0.],
                      [(self.dt * np.cos(z[4])) / self.m, (self.dt * np.cos(z[4])) / self.m],
                      [0., 0.],
                      [self.r * self.dt / self.I, -self.r * self.dt / self.I]])
        return A, B

    def solve_iLQR_trajectory(self, A, B, Q, q, R, r):
        # K, k
        K_gains = [0] * self.N
        k_feedforward = [0] * self.N
        # Pn, pn
        Pn = [0] * (self.N + 1)
        pn = [0] * (self.N + 1)
        Pn[self.N] = Q[-1]
        pn[self.N] = q[-1]
        for i, (An, Bn, Qn, qn, Rn, rn) in reversed(list(enumerate(zip(A, B, Q, q, R, r)))):
            K_gains[i] = -np.linalg.inv(Rn + (Bn.transpose() @ Pn[i + 1] @ Bn)) @ (Bn.transpose() @ Pn[i + 1] @ An)
            Pn[i] = Qn + (An.transpose() @ Pn[i + 1] @ An) + (An.transpose() @ Pn[i + 1] @ Bn @ K_gains[i])
            k_feedforward[i] = -np.linalg.inv(Rn + (Bn.transpose() @ Pn[i + 1] @ Bn)) @ ((Bn.transpose() @ pn[i + 1]) + rn)
            pn[i] = qn + (An.transpose() @ pn[i + 1]) + (An.transpose() @ Pn[i + 1] @ Bn @ k_feedforward[i])
        return K_gains, k_feedforward

    def line_search(self, J, K, k):
        if J > self.prevCost:
            self.alpha = self.alpha / 2
            if self.alpha < 0.01:
                return self.alpha
            for i, (state, control) in enumerate(zip(self.xstar.transpose(), self.ustar.transpose())):
                state, control = state.reshape((6, 1)), control.reshape((2, 1))
                self.ustar[:, i] = ((K[i] @ state) + (self.alpha * k[i]) + np.array([[self.m * self.g / 2], [self.m * self.g / 2]])).reshape((2, ))
                self.xstar[:, i + 1] = quadrotor.get_next_state(state.reshape((6, )), self.ustar[:, i].reshape((2, )))
            newCurrCost = self.compute_cost(self.xstar, self.ustar)
            return self.line_search(newCurrCost, K, k)
        else:
            alpha = self.alpha
            self.alpha = 1
            return alpha







    def vertical_orientation_controller(self, state, i):
        # 2. Tìm chi phí lớn nhất (đại diện cho bước tệ nhất) - phục vụ duy nhất cho "line Search"
        J = self.compute_cost(self.xstar, self.ustar)
        
        # 3. Các đại lượng đặc trưng của việc xấp xỉ (hàm chi phí bậc 2)
        An, Bn, Qn, qn, Rn, rn = self.get_quadratic_approximation_cost(self.xstar, self.ustar)
        
        # 4 backward Riccati equation -> K, k 
        K, k = self.solve_iLQR_trajectory(An, Bn, Qn, qn, Rn, rn)
        
        # 5. line search
        alpha = self.line_search(J, K, k)  # local optimization
        self.prevCost = J
        
        # => back to (2) until convergence 
        
        return ((K[i] @ state).reshape((2, 1)) + (alpha * k[i]) + np.array([[self.m * self.g / 2], [self.m * self.g / 2]])).reshape((2, ))