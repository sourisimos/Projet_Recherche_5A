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
        # FIXED :

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
            border = generate_hypercube_vertices(input_dim)
            self.points = np.concatenate((centered_points, border), axis=0)
            tot_num_points = num_points + 2**(input_dim)

        else:
            self.points = centered_points
            tot_num_points = num_points
        
        # Fonction plane
        self.points = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0, 1.0]])



        # Fonction avec une cassure
        # self.points =  np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0, 1.0], [0.0,0.5], [1.0, 0.5]])

        # Generating regions
        self.regions = Delaunay(self.points)

        # Génération des valeurs de sortie (dimension output_dim) pour chaque sommet
        self.values_at_vertices = np.random.rand(tot_num_points, output_dim)

        # FIXED
        self.values_at_vertices = np.array([[0.91233996],
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
        
        # self.values_at_vertices = (np.sum(self.points, axis=1)/2)**2
        # self.values_at_vertices = np.exp(np.sum(self.points, axis=1)**2)/ 10

        # self.values_at_vertices = self.values_at_vertices.reshape((tot_num_points,1))

        self.values_at_vertices = np.array([[0.1],[0.1],[0.7],[0.7]])

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

    def plot_2D(self, fig=None):
        if self.input_dim > 2 or self.output_dim != 1:
            print("La visualisation 3D est uniquement possible pour un maillage en entrée 2D "
                  "et des valeurs de sortie 1D !")

        else:
            if fig is None:
                fig = go.Figure()

            x = self.points[:, 0]
            y = self.points[:, 1]
            
            edges = set()
            for simplex in self.regions.simplices:
                edges.update([
                    (simplex[0], simplex[1]),
                    (simplex[1], simplex[2]),
                    (simplex[2], simplex[0])
                ])

            # Tracé des arêtes sur le plan z=0
            for edge in edges:
                fig.add_trace(go.Scatter3d(
                    x=[x[edge[0]], x[edge[1]]],
                    y=[y[edge[0]], y[edge[1]]],
                    z=[0, 0],  # Fixer z=0 pour la projection
                    mode='lines',
                    line=dict(
                        color='red',
                        width=3
                    ),
                    showlegend=False,
                ))

        return fig


    def plot_3D(self, fig=None):
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
