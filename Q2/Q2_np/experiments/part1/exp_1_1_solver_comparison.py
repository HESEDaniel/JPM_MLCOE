"""Linear solver comparison experiment."""
import numpy as np
from scipy import linalg as sla

def test_non_spd_matrix():
    """Test solvers on non-SPD matrix S."""

    # Non-SPD matrix (symmetric but has negative eigenvalue)
    S = np.array([[1.0, 2.0],
                  [2.0, 1.0]])
    b = np.array([1.0, 1.0])

    eigvals = np.linalg.eigvalsh(S)
            

    # np.linalg.inv
    try:
        x = np.linalg.inv(S) @ b
    except Exception as e:
        print(f"{'np.linalg.inv':<20} FAIL: {e}")

    # np.linalg.solve
    try:
        x = np.linalg.solve(S, b)
    except Exception as e:
        print(f"{'np.linalg.solve':<20} FAIL: {e}")

    # scipy.linalg.cholesky
    try:
        L = sla.cholesky(S, lower=True)
        x = sla.cho_solve((L, True), b)
    except Exception:
        pass

if __name__ == "__main__":
    test_non_spd_matrix()
