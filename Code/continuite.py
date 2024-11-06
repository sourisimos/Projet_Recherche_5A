import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point, Polygon

def generate_points(num_points):
    """Générer des points de contrôle aléatoires."""
    points = np.random.rand(num_points, 2) * 10  # Échelle de 0 à 10
    return points

def generate_piecewise_surface(points):
    """Générer une surface linéaire par morceaux à partir des points de contrôle."""
    # Trianguler les points
    tri = Delaunay(points)
    surfaces = []

    # Coefficients de fonction affine pour chaque polygone
    for simplex in tri.simplices:
        # Obtenir les sommets du triangle
        triangle = points[simplex]
        # Générer des coefficients aléatoires pour la fonction affine
        a = np.random.uniform(-1, 1)
        b = np.random.uniform(-1, 1)
        c = np.random.uniform(-5, 5)
        
        # Ajouter le triangle et ses coefficients
        surfaces.append((triangle, a, b, c))
        
        # Créer un quadrilatère à partir de deux triangles adjacents si possible
        if len(surfaces) > 1:
            prev_triangle = surfaces[-2][0]
            # Former un quadrilatère en utilisant les points du triangle courant et du triangle précédent
            quad = np.array([triangle[0], triangle[1], prev_triangle[1], prev_triangle[0]])
            surfaces.append((quad, a, b, c))  # Utiliser les mêmes coefficients pour simplicité

    return surfaces

def evaluate_surface(surfaces, x, y):
    """Évaluer la surface à un point donné (x, y)."""
    for polygon, a, b, c in surfaces:
        if is_point_in_polygon(x, y, polygon):
            return a * x + b * y + c
    return None

def is_point_in_polygon(x, y, polygon):
    """Vérifier si un point est à l'intérieur d'un polygone."""
    poly = Polygon(polygon)
    return poly.contains(Point(x, y))

def visualize_surface(points, surfaces):
    """Visualiser la nappe 2D en utilisant les surfaces."""
    # Créer une grille de points
    x_vals = np.linspace(0, 10, 100)
    y_vals = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    # Évaluer la surface sur la grille
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = evaluate_surface(surfaces, X[i, j], Y[i, j])

    # Tracer la surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Continuous Piecewise Surface with Various Polygons')
    plt.show()

# Paramètres
num_points = 50  # Augmenter le nombre de points pour plus de diversité

# Génération des points de contrôle
points = generate_points(num_points)

# Génération de la surface avec des polygones variés
surfaces = generate_piecewise_surface(points)

# Visualisation de la nappe
visualize_surface(points, surfaces)
