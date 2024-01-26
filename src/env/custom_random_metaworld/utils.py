import numpy as np

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])

def rotate_vector_by_quaternion(vector, quaternion):
    quaternion = np.array(quaternion) / np.linalg.norm(quaternion)
    vector = np.array(vector)

    quaternion_conjugate = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])

    rotated_vector_quaternion = quaternion_multiply(quaternion, quaternion_multiply([0] + vector.tolist(), quaternion_conjugate))
    rotated_vector = rotated_vector_quaternion[1:]

    return rotated_vector
