import torch

def kkt(p1_tilde, p2_tilde, lambda_1_tilde, lambda_2_tilde, q_pos1, q_theta1):
        rot_mat = torch.Tensor([[torch.cos(q_theta), -torch.sin(q_theta)],
                                [torch.sin(q_theta), torch.cos(q_theta)]])
        rot_mat_inv = torch.linalg.inv(rot_mat)
        box1_A = A_1_canonical @ rot_mat_inv
        box1_b = b_1_canonical + A_1_canonical @ rot_mat_inv @ q_pos1
        D = 2
        K = torch.zeros(2*D + num_1_points + num_2_points)
        # lagrangian gradient wrt p1
        K[:D] = 2* (p1_tilde - p2_tilde) + box1_A.T @ lambda_1_tilde
        K[D:2*D] = -2* (p1_tilde - p2_tilde) + A_2_canonical.T @ lambda_2_tilde
        K[2*D:2*D+num_1_points] = lambda_1_tilde * (box1_A @ p1_tilde - box1_b)
        K[2*D+num_1_points:2*D+num_1_points+num_2_points] = lambda_2_tilde * (A_2_canonical @ p2_tilde - b_2_canonical)
        return K

J_q_k_out = jacrev(lambda q_pos1, q_theta1: kkt(p1_tilde, p2_tilde, lambda_1_tilde, lambda_2_tilde, q_pos1, q_theta1), argnums=(0, 1))(q_pos, q_theta)