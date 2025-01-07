import numpy as np
import itertools
from tools import generate_hypercube_vertices


################
def affine_f(input_dim=2, output_dim=1, fixed= True):
    points = list(itertools.product([0, 1], repeat=input_dim))


    if fixed:
        print(f'ATTENTION: la fonction est de R² dans R !')
        values_at_vertices = np.array([[0.1], [0.1], [0.7], [0.7]])

    else: 

        A = np.random.uniform(-1, 1, size=(output_dim, input_dim))
        
        # Calcul initial des valeurs (sans b)
        values_no_bias = points @ A.T
        
        # Trouver les min et max des valeurs
        min_val = np.min(values_no_bias)
        max_val = np.max(values_no_bias)

        b_min = max(0, -min_val)
        b_max = min(1, 1 - max_val)


        b = np.random.uniform(b_min, b_max, size=(output_dim,))

        values_at_vertices = values_no_bias + b


    return (np.array(points), values_at_vertices)



def random_f(num_points, input_dim=2, output_dim=1, generate_cube=True):
    centered_points = np.random.rand(num_points, input_dim)
    if generate_cube : 
        border = generate_hypercube_vertices(input_dim)
        points = np.concatenate((centered_points, border), axis=0)
        tot_num_points = num_points + 2**(input_dim)

    else:
        points = centered_points
        tot_num_points = num_points

    values_at_vertices = np.random.rand(tot_num_points, output_dim)


    return np.array(points), np.array(values_at_vertices)


def fixed_f_2D(generate_cube=True):

    centered_points = np.array([[0.1230874,  0.9919432 ],
                                [0.24929276, 0.42547969],
                                [0.15300704, 0.73085746],
                                [0.35825542, 0.28526748],
                                [0.93157156, 0.73225214],
                                [0.90939576, 0.08385462],
                                [0.46912737, 0.55427293],
                                [0.44001697, 0.88191464],
                                [0.6207378,  0.30761995],
                                [0.03693253, 0.23952164]])

    if generate_cube : 
        border = generate_hypercube_vertices(2)
        points = np.concatenate((centered_points, border), axis=0)
        values_at_vertices = np.array([[0.91233996],
                                        [0.5314544 ],
                                        [0.27726649],
                                        [0.88542493], 
                                        [0.99150616], 
                                        [0.29222178], 
                                        [0.0054213 ], 
                                        [0.83149493], 
                                        [0.74419698], 
                                        [0.90944431], 
                                        [0.12867826], 
                                        [0.86673492], 
                                        [0.72346853],
                                        [0.11678231]])

    else:
        points = centered_points
        values_at_vertices = np.array([[0.91233996],
                                        [0.5314544 ],
                                        [0.27726649],
                                        [0.88542493], 
                                        [0.99150616], 
                                        [0.29222178], 
                                        [0.0054213 ], 
                                        [0.83149493], 
                                        [0.74419698], 
                                        [0.90944431]])

    return np.array(points), np.array(values_at_vertices)




def heaviside_f_2D(steep= 2):
    if steep <= 1:
        raise ValueError("La valeur de pente doit être strictement supérieur à 1!")

    border = list(itertools.product([0, 1], repeat=2))

    b = 0.5 * (1 - steep)

    val_0 = -b/steep # Avoir la valeur d'intersection en la pente en 0 et la focntion affine centrale
    val_1 = (1-b)/steep # Avoir la valeur d'intersection en la pente en 1 et la focntion affine centrale

    heavi_points = [[0,val_0], [1, val_0], [0,val_1], [1, val_1] ]

    points = np.concatenate((border, heavi_points), axis=0)

    values_at_vertices = [[0.], [1.], [0.], [1.],
                          [0.], [0.], [1.], [1.]]


    return np.array(points), np.array(values_at_vertices)


def cosine_f_2D(precisionx=10, previsiony=2):
    """
    divide space into a precisionx * precisiony space 
    """

    x_abs, y_abs = np.linspace(0, 1, precisionx), np.linspace(0, 1, previsiony)
    X, Y = np.meshgrid(x_abs, y_abs)
    Z = np.cos(X * 3 * np.pi) / 4 + 0.5

    points = np.column_stack((X.ravel(), Y.ravel()))
    values_at_vertices = np.reshape(Z, [-1,1])

    return points, values_at_vertices

