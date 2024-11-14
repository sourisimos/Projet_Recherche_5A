import itertools
import numpy as np



def generate_hypercube_vertices(n):
    """
    Genere tous les sommets du cube de dimension n. 
    input : - n (int) : nb de dimension des données d'entrée

    output : (nd.array) each coordinates of the vertices
    
    """
    vertices = list(itertools.product([0, 1], repeat=n))
    # Convertir en tableau numpy si nécessaire
    return np.array(vertices)