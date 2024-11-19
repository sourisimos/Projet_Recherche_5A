import itertools
import numpy as np
from scipy.spatial import HalfspaceIntersection
import numpy as np
import plotly.graph_objects as go
import random

def generate_hypercube_vertices(n):

    vertices = list(itertools.product([0, 1], repeat=n))
    # Convertir en tableau numpy si nécessaire
    return np.array(vertices)

def plot_affine_zones_with_meshgrid_3D(fig, zones_affines, constraints, x_range=(0, 1), y_range=(0, 1), resolution=100):
    c=0
    # Créer un meshgrid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    for idx, (pattern, _) in enumerate(zones_affines.items()):

        w = np.array(constraints[pattern]['weights'])  # Poids
        b = np.array(constraints[pattern]['biases'])   # Biais
        region_points = []

        # Vérifier quelles contraintes sont satisfaites
        for pt in grid_points:
            if (np.dot(w, pt) + b <= 0).all() == True:
                region_points.append(pt)
        vx = [v[0] for v in region_points]
        vy = [v[1] for v in region_points]
        c += len(region_points)
        z = random.uniform(-1, 0)

        # Ajouter la région comme surface sur la figure
        if len(region_points) > 0:
            fig.add_trace(go.Mesh3d(
                x=vx,
                y=vy,
                z=[-0.05*idx] * len(region_points),  # Fixer z=0 pour affichage 3D 
                color=f'rgb({np.random.randint(50, 255)}, {np.random.randint(50, 255)}, {np.random.randint(50, 255)})',
                opacity=1,
                name=f"Région - {pattern}"
            ))
    print("Nombre de zones générées; ", len(zones_affines))
    print("Nombre de points dans les zones",c,'/',resolution**2)
    return None