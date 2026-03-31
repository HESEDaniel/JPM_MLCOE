"""RKHS Particle Flow Filter."""
import numpy as np


def localization_matrix(n_x, r_in=4.0):
    """Gaussian localization: C[i,j] = exp(-((i-j)/r_in)^2)."""
    idx = np.arange(n_x)
    return np.exp(-((idx[:, None] - idx[None, :]) / r_in)**2)


def rkhs_particle_flow(particles, h, H_jac, R, y, n_steps=10, step_size=0.1,
                       loc_radius=4.0, bandwidth_alpha=None, kernel_type='matrix-valued',
                       store_history=False, adaptive_step=False, step_factor=1.4,
                       decrease_patience=20, min_step=1e-6, max_step=1.0):
    """
    RKHS particle flow (Hu21, Algorithm 1-2).

    Parameters
    ----------
    particles : ndarray [N, n_x]
    h, H_jac : callable
        Observation function and its Jacobian
    R : ndarray [n_y, n_y]
        Observation noise covariance
    y : ndarray [n_y]
    n_steps : int
    step_size : float
        Pseudo time step
    loc_radius : float
        Localization radius r_in
    bandwidth_alpha : float
        Bandwidth scaling (default: 1/N)
    kernel_type : str
        'scalar' or 'matrix-valued'
    store_history : bool
    adaptive_step : bool
        Adaptive step size (Hu21 Section 3.2)
    step_factor, decrease_patience, min_step, max_step : float
        Adaptive step parameters

    Returns
    -------
    particles : ndarray [N, n_x]
    (particles, flow_history, diagnostics) : tuple if store_history=True
    """
    N, n_x = particles.shape
    y = np.atleast_1d(y)

    # Ensemble covariance with localization
    x_bar = np.mean(particles, axis=0)
    X = (particles - x_bar).T
    B = X @ X.T / (N - 1)
    C = localization_matrix(n_x, loc_radius)
    B_loc = B * C
    B_loc_inv = np.linalg.inv(B_loc)
    R_inv = np.linalg.inv(R)

    alpha = bandwidth_alpha if bandwidth_alpha is not None else 1.0 / N

    if kernel_type == 'scalar':
        A = np.linalg.inv(alpha * B_loc)
        bandwidths = None
    else:
        bandwidths = alpha * np.diag(B_loc)
        A = None

    # History tracking
    if store_history:
        flow_history = [particles.copy()]
        flow_magnitudes = []
        step_sizes = []

    # Adaptive step state
    current_step = step_size
    prev_flow_mag = np.inf
    consecutive_decreases = 0

    for _ in range(n_steps):
        # Log-posterior gradient
        grad_log_post = np.array([
            H_jac(particles[i]).T @ R_inv @ (y - h(particles[i])) - B_loc_inv @ (particles[i] - x_bar)
            for i in range(N)
        ])

        if kernel_type == 'scalar':
            K = np.zeros((N, N))
            div_K = np.zeros((N, N, n_x))
            for i in range(N):
                diff = particles - particles[i]
                quad = np.sum((diff @ A) * diff, axis=1)
                K[i, :] = np.exp(-0.5 * quad)
                div_K[i, :, :] = -K[i, :, None] * (diff @ A.T)

            # Stein operator
            I_flow = np.zeros((N, n_x))
            for i in range(N):
                term1 = K[i, :] @ grad_log_post
                term2 = np.sum(div_K[i, :, :], axis=0)
                I_flow[i] = (term1 + term2) / N
        else:
            # Matrix-valued kernel
            I_flow = np.zeros((N, n_x))
            for d in range(n_x):
                diff = particles[:, d, None] - particles[:, d]
                K_d = np.exp(-diff**2 / (2 * bandwidths[d]))
                grad_K_d = diff / bandwidths[d] * K_d
                term1 = K_d @ grad_log_post[:, d]
                term2 = np.sum(grad_K_d, axis=1)
                I_flow[:, d] = (term1 + term2) / N

        flow = I_flow @ B_loc.T
        flow_mag = np.mean(np.linalg.norm(flow, axis=1))

        # Adaptive step size
        if adaptive_step:
            if flow_mag < prev_flow_mag:
                consecutive_decreases += 1
                if consecutive_decreases >= decrease_patience:
                    current_step = min(current_step * step_factor, max_step)
                    consecutive_decreases = 0
            else:
                current_step = max(current_step / step_factor, min_step)
                consecutive_decreases = 0
            prev_flow_mag = flow_mag

        particles = particles + current_step * flow

        if store_history:
            flow_history.append(particles.copy())
            flow_magnitudes.append(flow_mag)
            step_sizes.append(current_step)

    if store_history:
        diagnostics = {
            'flow_magnitudes': np.array(flow_magnitudes),
            'kernel_type': kernel_type,
            'bandwidth_alpha': alpha,
            'n_steps': n_steps,
            'initial_step_size': step_size,
            'final_step_size': current_step,
            'adaptive_step': adaptive_step,
            'step_sizes': np.array(step_sizes) if step_sizes else None,
        }
        return particles, flow_history, diagnostics

    return particles


def rkhs_particle_flow_filter(f, h, H_jac, Q, R, m0, P0, ys,
                               N_particles=100, n_flow_steps=10, step_size=0.1,
                               loc_radius=4.0, bandwidth_alpha=None,
                               kernel_type='matrix-valued', rng=None,
                               adaptive_step=False, step_factor=1.4,
                               decrease_patience=20):
    """
    RKHS Particle Flow Filter for sequential state estimation.

    Parameters
    ----------
    f, h, H_jac : callable
        Transition, observation, and observation Jacobian
    Q, R : ndarray
        Process and observation noise covariances
    m0, P0 : ndarray
        Initial mean and covariance
    ys : ndarray [T, n_y]
    N_particles, n_flow_steps, step_size, loc_radius, bandwidth_alpha, kernel_type
    adaptive_step, step_factor, decrease_patience
        See rkhs_particle_flow

    Returns
    -------
    m_filt : ndarray [T, n_x]
    P_filt : ndarray [T, n_x, n_x]
    ess_history : ndarray [T]
        Always N (equal weights after flow)
    resample_count : int
        Always 0 (no resampling needed)
    """
    if rng is None:
        rng = np.random.default_rng()

    T, n_x, N = len(ys), len(m0), N_particles
    particles = rng.multivariate_normal(m0, P0, size=N)

    m_filt = np.zeros((T, n_x))
    P_filt = np.zeros((T, n_x, n_x))
    ess_history = np.full(T, N, dtype=float)

    for t in range(T):
        y_t = np.atleast_1d(ys[t])

        # Prediction
        particles = np.array([rng.multivariate_normal(f(particles[i]), Q) for i in range(N)])

        # RKHS flow update
        particles = rkhs_particle_flow(
            particles, h, H_jac, R, y_t,
            n_steps=n_flow_steps, step_size=step_size,
            loc_radius=loc_radius, bandwidth_alpha=bandwidth_alpha,
            kernel_type=kernel_type,
            adaptive_step=adaptive_step, step_factor=step_factor,
            decrease_patience=decrease_patience
        )

        # Posterior statistics (equal weights)
        m_filt[t] = np.mean(particles, axis=0)
        diff = particles - m_filt[t]
        P_filt[t] = np.einsum('ij,ik->jk', diff, diff) / (N - 1)

    return m_filt, P_filt, ess_history, 0


def rkhs_pff_linear_gaussian(ys, A, H, Q, R, m0, P0, N_particles=100,
                              n_flow_steps=10, step_size=0.1, loc_radius=4.0, rng=None):
    """RKHS PFF for linear Gaussian model."""
    return rkhs_particle_flow_filter(
        f=lambda x: A @ x, h=lambda x: H @ x, H_jac=lambda x: H,
        Q=Q, R=R, m0=m0, P0=P0, ys=ys,
        N_particles=N_particles, n_flow_steps=n_flow_steps,
        step_size=step_size, loc_radius=loc_radius, rng=rng
    )


def rkhs_pff_nonlinear(ys, f, h, H_jac, Q, R, m0, P0, N_particles=100,
                        n_flow_steps=10, step_size=0.1, loc_radius=4.0, rng=None):
    """RKHS PFF for nonlinear model."""
    return rkhs_particle_flow_filter(
        f, h, H_jac, Q, R, m0, P0, ys,
        N_particles=N_particles, n_flow_steps=n_flow_steps,
        step_size=step_size, loc_radius=loc_radius, rng=rng
    )
