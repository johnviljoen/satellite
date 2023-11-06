import casadi as ca
import numpy as np
import utils.pytorch as ptu
from utils.orbital_elements import state_to_orbital_elements
from utils.rotation import quaternion_error
from dynamics import state_dot

class MPC:
    def __init__(
            self,
            N,                          # Prediction Horizon No. Inputs
            Ts_0,                       # Timestep of Simulation
            Tf_hzn,                     # Final time prediction horizon reaches
            dts_init,                   # initial variable timestep
            state_dot,                  # CasADI dynamics
            init_state,                 # where to do first solve to start warm start variables

            integrator_type = "euler",  # "euler", "RK4"
        ):

        self.N = N
        self.Ts = Ts_0
        self.Tf_hzn = Tf_hzn
        self.dts_init = dts_init
        self.dynamics = state_dot
        self.state_ub = None
        self.state_lb = None
        self.integrator_type = integrator_type

        # 13 free space states, 6 orbital elements, 6 acceleration inputs
        self.nx, self.noe, self.m = 13, 6, 6

        # create optimizer and define its optimization variables
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx + self.noe, N+1)
        self.U = self.opti.variable(self.m, N+1) # final input plays no role

        # adaptive timestep means that it must be seen as a parameter to the opti
        self.dts = self.opti.parameter(self.N)
        self.opti.set_value(self.dts, self.dts_init)

        # create Q, R
        self.Q, self.R = self.create_weight_matrices()

        # apply the dynamics constraints over timesteps defined in dts
        self.apply_dynamics_constraints(self.opti, self.dts)

        # apply the state and input constraints
        self.apply_state_input_constraints(self.opti)

        # solver setup
        opts = {
            'ipopt.print_level':0, 
            'print_time':0,
            'ipopt.tol': 1e-6,
        } # silence!
        self.opti.solver('ipopt', opts)

        # define start condition (dummy)
        self.init = self.opti.parameter(self.n,1)
        self.opti.set_value(self.init, init_state)
        self.opti.subject_to(self.X[:,0] == self.init)

        # define dummy reference (n x N)
        reference = np.zeros([self.n,N+1])
        self.ref = self.opti.parameter(self.n,N+1)
        self.opti.set_value(self.ref, reference)

        # cost function
        self.opti.minimize(self.cost(self.X, self.ref, self.U)) # discounted

        # solve the mpc once, so that we can do it repeatedly in a method later
        sol = self.opti.solve()

        # use the initial solution as the first warm start
        self.x_sol, self.u_sol = sol.value(self.X), sol.value(self.U)


    def __call__(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        # point to point control, from state to stationary reference

        # define start condition
        self.opti.set_value(self.init, state)

        # calculate the adaptive dts
        dts = self.adaptive_dts_Tf_hzn(state, reference)
        self.opti.set_value(self.dts, dts)

        # warm starting
        # reference_stack = np.array([np.linspace(state[i], reference[i,-1], self.N) for i in range(self.n)])
        old_x_sol = self.x_sol[:,2:] # ignore old start and first step (this step start)
        x_warm_start = np.hstack([old_x_sol, old_x_sol[:,-1:]]) # stack final solution onto the end again for next warm start
        old_u_sol = self.u_sol[:,1:] # ignore previous solution
        u_warm_start = np.hstack([old_u_sol, old_u_sol[:,-1:]]) # stack final u solution onto the end again for next warm start

        self.opti.set_initial(self.X[:,1:], x_warm_start)
        self.opti.set_initial(self.U[:,:], u_warm_start)

        # define cost w.r.t reference
        self.opti.set_value(self.ref, reference)

        # solve
        sol = self.opti.solve()

        # save the solved x's and u'x to warm start the next time around
        self.x_sol, self.u_sol = sol.value(self.X), sol.value(self.U)

        # return first input to be used
        return self.u_sol[:,0]

    def cost(self, state, reference, input):

        # state is {x, y, z, q0, q1, q2, q3, xdot, ydot, zdot, p, q, r}
        # reference is {a, e, i, Omega, omega, nu, [quat_des]}
        #               |--- orbital elements ---|-- quat ---|

        q_err = quaternion_error.casadi(state[3:7,:], reference[6:,:])

        state_error = reference - state
        cost = ca.MX(0)
        # lets get cost per timestep:
        for k in range(self.N + 1):
            timestep_input = input[:,k]
            timestep_orbital_elements = state_to_orbital_elements.casadi(state[:,k])
            timestep_oe_error = reference - timestep_orbital_elements
            cost += (timestep_oe_error.T @ self.Q @ timestep_oe_error + timestep_input.T @ self.R @ timestep_input)
        return cost

    # Constraints and Weights Methods 
    # -------------------------------

    def create_weight_matrices(self):

        # define weighting matrices
        Q = ca.MX.zeros(self.nx + self.noe, self.nx + self.noe)
        Q[0,0] =   0 # x
        Q[1,1] =   0 # y
        Q[2,2] =   0 # z
        Q[3,3] =   1 # q0
        Q[4,4] =   1 # q1
        Q[5,5] =   1 # q2
        Q[6,6] =   1 # q3
        Q[7,7] =   0 # xdot
        Q[8,8] =   0 # ydot
        Q[9,9] =   0 # zdot
        Q[10,10] = 1 # p
        Q[11,11] = 1 # q
        Q[12,12] = 1 # r
        Q[13,13] = 1 # a
        Q[14,14] = 1 # e
        Q[15,15] = 1 # i
        Q[16,16] = 1 # Omega
        Q[17,17] = 1 # omega
        Q[18,18] = 1 # nu

        R = ca.MX.eye(self.m)

        return Q, R

    def apply_dynamics_constraints(self, opti, dts):
        # constrain optimisation to the system dynamics over the horizon
        if self.integrator_type == 'euler':
            for k in range(self.N):
                input = self.U[:,k]
                sdot_k = self.dynamics(self.X[:self.nx,k], input)
                x_next = self.X[:self.nx,k] + sdot_k * dts[k]
                opti.subject_to(self.X[:self.nx,k+1] == x_next)
                self.apply_orbital_elements_constraints(opti, x_next, k)

        elif self.integrator_type == 'RK4':
            for k in range(self.N):
                k1 = self.dynamics(self.X[:self.nx,k], self.U[:,k])
                k2 = self.dynamics(self.X[:self.nx,k] + dts[k] / 2 * k1, self.U[:,k])
                k3 = self.dynamics(self.X[:self.nx,k] + dts[k] / 2 * k2, self.U[:,k])
                k4 = self.dynamics(self.X[:self.nx,k] + dts[k] * k3, self.U[:,k])
                x_next = self.X[:self.nx,k] + dts[k] / 6 * (k1 + 2*k2 + 2*k3 + k4)
                opti.subject_to(self.X[:self.nx,k+1] == x_next)
                self.apply_orbital_elements_constraints(opti, x_next, k)

    def apply_orbital_elements_constraints(self, opti, x_next, k):
        next_orbital_elements = state_to_orbital_elements.casadi(x_next)
        opti.subject_to(self.X[self.nx:,k+1] == next_orbital_elements)

    def apply_state_input_constraints(self, opti):
        # apply state constraints
        for k in range(self.N):
            opti.subject_to(self.X[:,k] < self.state_ub)
            opti.subject_to(self.X[:,k] > self.state_lb)

        # define input constraints
        opti.subject_to(opti.bounded(-100, self.U, 100))

    # Utility Methods
    # ---------------

    def get_predictions(self):
        return self.opti.value(self.X), self.opti.value(self.U)
    
    def dist2point(self, state, point):
        return np.sqrt((state[0] - point[0])**2 + (state[1] - point[1])**2 + (state[2] - point[2])**2)
    
    def adaptive_dts_Tf_hzn(self, state, reference):

        dist2end = self.dist2point(state, reference[0:3,-1])
        unit_distance = 3 # 3 # np.sqrt(2**2 + 2**2 + 1**2)
        offset = 3.0
        # reduce Tf_hzn as distance to point decreases
        Tf_hzn = self.Tf_hzn #* ((dist2end + offset) / (unit_distance + offset))
        # Find the optimal dts for the MPC
        d = (2 * (Tf_hzn/self.N) - 2 * self.Ts) / (self.N - 1)
        dts = [self.Ts + i * d for i in range(self.N)]

        # print(f"dts_delta: {dts[-1] - dts[0]}")
        # print(f"lookahead time: {sum(dts)}")

        return dts
    
if __name__ == "__main__":

    from params import get_params

    params = get_params()

    Ti = 0.0
    Ts = 1.0
    Tf = 10000
    Tf_hzn = 1000
    N = 10

    # Find the optimal dts for the MPC
    dt_1 = Ts
    d = (2 * (Tf_hzn/N) - 2 * dt_1) / (N - 1)
    dts_init = [dt_1 + i * d for i in range(N)]

    mpc = MPC(
        N=N,
        Ts_0=Ts,
        Tf_hzn=Tf_hzn,
        dts_init=dts_init,
        state_dot=state_dot.casadi,
        init_state=params["init_state"],
        integrator_type='euler'
    )

