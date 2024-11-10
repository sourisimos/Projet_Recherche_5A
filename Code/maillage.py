import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class MaillageDelaunayMultiDimension:
    def __init__(self, num_points=100, input_dim=2, output_dim=1):
        """
        Initialise un maillage Delaunay dans un espace de dimension arbitraire.

        num_points : Nombre de points à générer pour le maillage
        input_dim : Dimension de l'espace du maillage d'entrée (par exemple, 2 ou 3)
        output_dim : Dimension de l'espace des valeurs interpolées de sortie
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.points = np.random.rand(num_points, input_dim)
        self.regions = Delaunay(self.points)
        # Génération des valeurs de sortie (dimension output_dim) pour chaque sommet
        self.values_at_vertices = np.random.rand(num_points, output_dim)

    def evaluate_function_at_point(self, point):
        """
        Évalue la fonction interpolée en un point donné.

        point : np.array de taille (input_dim,) - coordonnées du point à évaluer.

        Retourne : un tableau numpy de dimension (output_dim) contenant la valeur
                   interpolée de la fonction en ce point, ou None si en dehors du maillage.
        """
        simplex = self.regions.find_simplex(point)

        if simplex != -1:  # Vérifier que le point est dans le maillage
            simplex_vertices = self.regions.simplices[simplex]
            vertices = self.points[simplex_vertices]
            values = self.values_at_vertices[simplex_vertices]

            # Calcul des coordonnées barycentriques pour le point
            T = np.vstack((vertices.T, np.ones(len(simplex_vertices))))
            b = np.append(point, 1)
            bary_coords = np.linalg.solve(T, b)

            # Interpolation de la valeur de sortie pour chaque dimension de output_dim
            result = np.dot(bary_coords, values)
            print("Valeur de la fonction au point", point, ":", result)
            return result
        else:
            print("Le point est en dehors du maillage.")
            return None

    def plot(self):
        """
        Affiche le maillage Delaunay et une des dimensions de la fonction interpolée en 3D,
        si la dimension d'entrée est de 2 et la dimension de sortie est de 3.
        """
        if self.input_dim >= 2 or self.output_dim != 1:
            print("La visualisation 3D est uniquement possible pour un maillage en entrée 2D "
                  "et des valeurs de sortie 3D !")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Création d'une collection de triangles pour la surface 3D en utilisant la 3e dimension des valeurs
        for simplex in self.regions.simplices:
            vertices = self.points[simplex]
            # Mapping chaque sommet à sa coordonnée (x, y, z) interpolée
            reg_vertices = [(vertices[i][0], vertices[i][1], self.values_at_vertices[simplex[i], 0]) for i in range(3)]
            color_value = np.mean(self.values_at_vertices[simplex, 0])
            poly = Poly3DCollection([reg_vertices], color=plt.cm.viridis(color_value), edgecolor="k", alpha=0.7)
            ax.add_collection3d(poly)

        # Paramètres des axes
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Valeur de la fonction (3ème dimension de sortie)")
        ax.set_title("Visualisation 3D de la fonction interpolée sur le maillage")

        # Échelle de couleurs
        plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax, label="Valeur de la fonction", shrink=0.5, pad=0.1)
        plt.show()
