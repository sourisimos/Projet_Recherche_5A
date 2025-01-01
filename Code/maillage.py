import numpy as np
from scipy.spatial import Delaunay
import plotly.graph_objects as go


from objective_functions import affine_f, random_f, fixed_f_2D, heaviside_f_2D



class MaillageDelaunayMultiDimension:
    def __init__(self, num_points=10, input_dim=2, output_dim=1, generate_cube = True):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.generate_cube = generate_cube


        self.num_points = num_points


        # Get coords de la fonction objectif
        # (self.points, self.values_at_vertices) = affine_f(self.input_dim, self.output_dim, fixed = True)
        (self.points, self.values_at_vertices) = heaviside_f_2D(4)


        # Generating regions
        self.regions = Delaunay(self.points)


    def evaluate_function_at_point(self, point, text_disp = False):

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
