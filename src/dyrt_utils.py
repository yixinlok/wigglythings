import numpy as np
import globals 

def skew(v):
    """
    Return the skew-symmetric matrix of a 3D vector.
    v: array-like of shape (3,)
    """
    v = np.asarray(v).reshape(3,)
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def construct_big_gamma(v):
    """
    Return gamma matrix for a list of 3D points.
    v: array-like of shape (3,N)

    """
    gamma_list = []
    for i in range(v.shape[1]):
        x = np.hstack((-skew(v[:,i]), np.eye(3)))
        gamma_list.append(x)
    gamma = np.vstack(gamma_list)
    return gamma

def test_construct_big_gamma():
    print("testing construct_big_gamma...")
    v = np.array([[1,2,3],[7,8,9]]).T
    print("input v:")
    print(v)
    gamma = construct_big_gamma(v)
    print(gamma)
    print(gamma.shape)

def compute_IIR_params(w_vec, xi_vec=None):
    """
    Compute IIR filter parameters epsilon and gamma for given natural frequencies,
    damping ratios, and time step size.
    
    Parameters:
    w_vec : array-like
        Natural frequencies (rad/s) for each mode.
    xi_vec : array-like
        Damping ratios for each mode.

    Notes: 
    - when xi increases, the system is more damped
    - xi should be in (0, 1) for underdamped system, does not run otherwise
    """
    # if xi_vec is None:
    #     xi_vec = np.array([0.05,0.05,0.05,0.05,0.05,0.05])

    w_vec = np.where(w_vec == 0, 1e-6, w_vec)
    alpha = 0.1
    beta = 0.9
    xi_vec = 0.5*(alpha/w_vec + beta*w_vec)  # damping ratio
    xi_max = np.max(xi_vec)
    xi_vec = xi_vec/(xi_max + 0.03)  # normalize to max xi

    w_di_vec = w_vec * np.sqrt(1 - xi_vec**2)  
    theta_vec = w_di_vec * globals.TIME_STEP_SIZE                   
    epsilon_vec = np.exp(-xi_vec * w_vec * globals.TIME_STEP_SIZE)  
    gamma_vec = np.arcsin(xi_vec)
    assert w_vec.shape == w_di_vec.shape == theta_vec.shape == epsilon_vec.shape == gamma_vec.shape          

    first_term = 2*epsilon_vec*np.cos(theta_vec) 
    second_term = -epsilon_vec**2
    third_term_coeff_num = 2*(epsilon_vec*np.cos(theta_vec+gamma_vec)-(epsilon_vec**2)*np.cos(2*theta_vec+gamma_vec))
    third_term_coeff_denom = 3*w_vec*w_di_vec
    third_term = third_term_coeff_num / third_term_coeff_denom

    return first_term, second_term, third_term  


if __name__ == "__main__":
    # test different xi vec values
    compute_IIR_params(np.array([0.01, 0.05, 0.1, 0.2]))