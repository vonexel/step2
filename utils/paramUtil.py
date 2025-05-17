import numpy as np

# Define a kinematic tree for the skeletal structure (e.g., limbs and joints) of different datasets
# Used to transform 3D-motion into latent space tokens
kit_kinematic_chain = [
    [0, 11, 12, 13, 14, 15],  # Right leg chain
    [0, 16, 17, 18, 19, 20],  # Left leg chain
    [0, 1, 2, 3, 4],          # Torso and head chain
    [3, 5, 6, 7],             # Right arm chain
    [3, 8, 9, 10],            # Left arm chain
]
# Bone offsets in 3D space (x, y, z)
# KIT-Dataset
kit_raw_offsets = np.array(
    [
        [0, 0, 0],   # Root joint
        [0, 1, 0],   # Torso (upward)
        [0, 1, 0],   # Chest
        [0, 1, 0],   # Neck
        [0, 1, 0],   # Head
        [1, 0, 0],   # Right forearm
        [0, -1, 0],  # Right elbow
        [0, -1, 0],  # Right hand
        [-1, 0, 0],  # Left forearm
        [0, -1, 0],  # Left elbow
        [0, -1, 0],  # Left hand
        [1, 0, 0],   # Right hip
        [0, -1, 0],  # Right knee
        [0, -1, 0],  # Right foot
        [0, 0, 1],   # Right heel
        [0, 0, 1],   # Right toes
        [-1, 0, 0],  # Left hip
        [0, -1, 0],  # Left knee
        [0, -1, 0],  # Left foot
        [0, 0, 1],   # Left heel
        [0, 0, 1],   # Left toes
    ]
)
# Text-to-Motion (T2M) dataset
t2m_raw_offsets = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ]
)

t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],     # Right leg
    [0, 1, 4, 7, 10],     # Left leg
    [0, 3, 6, 9, 12, 15], # Torso and head
    [9, 14, 17, 19, 21],  # Right arm
    [9, 13, 16, 18, 20],  # Left arm
]
# Kinematic chains for hands
t2m_left_hand_chain = [
    [20, 22, 23, 24],
    [20, 34, 35, 36],
    [20, 25, 26, 27],
    [20, 31, 32, 33],
    [20, 28, 29, 30],
]
t2m_right_hand_chain = [
    [21, 43, 44, 45],
    [21, 46, 47, 48],
    [21, 40, 41, 42],
    [21, 37, 38, 39],
    [21, 49, 50, 51],
]

# Target skeleton IDs for datasets
kit_tgt_skel_id = "03950"

t2m_tgt_skel_id = "000021"
