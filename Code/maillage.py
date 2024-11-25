import numpy as np
from scipy.spatial import Delaunay
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tools import generate_hypercube_vertices

class MaillageDelaunayMultiDimension:
    def __init__(self, num_points=100, input_dim=2, output_dim=1, generate_cube = True):
        """
        Initialise un maillage Delaunay dans un espace de dimension arbitraire.

        num_points : Nombre de points à générer pour le maillage
        input_dim : Dimension de l'espace du maillage d'entrée (par exemple, 2 ou 3)
        output_dim : Dimension de l'espace des valeurs interpolées de sortie
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Generating points
        centered_points = np.random.rand(num_points, input_dim)
        
        if generate_cube : 
            border = generate_hypercube_vertices(input_dim)
            self.points = np.concatenate((centered_points, border), axis=0)
            tot_num_points = num_points + 2**(input_dim)

        else:
            self.points = centered_points
            tot_num_points = num_points

        # Generating regions
        self.regions = Delaunay(self.points)

        # Génération des valeurs de sortie (dimension output_dim) pour chaque sommet
        self.values_at_vertices = np.random.rand(tot_num_points, output_dim)

    def evaluate_function_at_point(self, point, text_disp = False):
        """
        Évalue la fonction interpolée en un point donné.

        point : np.array de taille (input_dim,) - coordonnées du point à évaluer.

        Retourne : - un tableau numpy de dimension (output_dim) contenant la valeur
                   interpolée de la fonction en ce point, ou None si en dehors du maillage.
                   - rien si le point est hors du maillage 
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

            if text_disp: 
                print("Valeur de la fonction au point", point, ":", result)

            return result
        else:
            print("Le point est en dehors du maillage.")

            return None

    def plot(self, fig=None):
        """
        Affiche le maillage Delaunay et une des dimensions de la fonction interpolée en 3D,
        si la dimension d'entrée est de 2 et la dimension de sortie est de 1.

        fig : Figure Plotly à enrichir (par défaut, crée une nouvelle figure)

        retourne : None
        """

        if self.input_dim > 2 or self.output_dim != 1:
            print("La visualisation 3D est uniquement possible pour un maillage en entrée 2D "
                  "et des valeurs de sortie 1D !")
        else:
            if fig is None:
                fig = go.Figure()

            # Extraction des coordonnées des trgl
            x = self.points[:, 0]
            y = self.points[:, 1]
            z = self.values_at_vertices[:, 0]

            # Extraction des indices des trgl
            i, j, k = zip(*self.regions.simplices)


            z_min, z_max = z.min(), z.max()
            intensity = (z - z_min) / (z_max - z_min)  # Entensité avec 0 et 1 comme optimum

            # Tracé
            fig.add_trace(go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                intensity=intensity, # intensité selon Z
                colorscale="Plasma", # Colormap pour Z
                opacity=0.4, # Opacité
                showscale=True, # Ajouter une barre d'échelle
                colorbar=dict( # mise en forme de la barre
                    title="Z<br>(Maillage)",
                    len=0.6,
                    x=-0.03,
                    xpad=40,
                    orientation="v"
                )
            ))
            return fig


    def generate_points_region(self, n=1):
        """
        Génère un nombre spécifié de points aléatoires dans chaque région (simplexe).

        num_points_per_region : nombre de points à générer dans chaque région.

        Retourne : Deux arrays, l'un avec les coordonnées, l'autre avec la valeur en ce point.
        """

        points_in_regions = []
        values_in_regions = []

        for simplex_index, simplex in enumerate(self.regions.simplices):
            vertices = self.points[simplex]

            for _ in range(n):
                # Générer des coordonnées barycentriques aléatoires qui somment à 1
                bary_coords = np.random.rand(len(vertices))
                bary_coords /= bary_coords.sum()  # Normalisation
                
                # Calculer le point dans l'espace en utilisant les coordonnées barycentriques
                random_point = np.dot(bary_coords, vertices)

                point_value = self.evaluate_function_at_point(random_point)

                points_in_regions.append(random_point)
                values_in_regions.append(point_value)

        return np.array(points_in_regions), np.array(values_in_regions)
