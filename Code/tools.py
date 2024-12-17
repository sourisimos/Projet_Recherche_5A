import itertools
import numpy as np
import plotly.graph_objects as go

from scipy.spatial import ConvexHull


def generate_hypercube_vertices(n):

    vertices = list(itertools.product([0, 1], repeat=n))
    # Convertir en tableau numpy si nécessaire
    return np.array(vertices)

def plot_affine_zones_with_meshgrid_2D(fig, zones_affines, constraints, x_range=(0, 1), y_range=(0, 1), resolution=50):
    c=0
    # Créer un meshgrid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    for i, pattern in enumerate(zones_affines.keys()):
        #   print(i/len(zones_affines.keys()))
        w = np.array(constraints[pattern]['weights'])  # Poids
        b = np.array(constraints[pattern]['biases'])   # Biais

        region_points = []


        for pt in grid_points:
            if (np.dot(w, pt) + b <= 0).all():
                region_points.append(pt)
        

        vx = [v[0] for v in region_points]
        vy = [v[1] for v in region_points]
        c += len(region_points)

        # Ajouter la région comme surface sur la figure
        if len(region_points) > 0:
            fig.add_trace(go.Mesh3d(
                x=vx,
                y=vy,
                z=[0] * len(region_points),  # Fixer z=0 pour affichage 3D 
                color=f'rgb({np.random.randint(50, 255)}, {np.random.randint(50, 255)}, {np.random.randint(50, 255)})',
                opacity=1,
            ))
    print("Nombre de zones générées; ", len(zones_affines))
    print("Nombre de points dans les zones",c,'/',resolution**2)

    return fig.data



def plot_affine_zones_with_meshgrid_2D_border(fig, zones_affines, constraints, x_range=(0, 1), y_range=(0, 1), resolution=1000):
    if resolution < 1000:
        print('Attenion, une résolution inférieur à 1000 peut ne pas fonctionner !')
    # Créer un meshgrid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # Points du maillage

    c = 0  # Compteur de points dans les zones
    i = 0
    for i, (pattern, _) in enumerate(zones_affines.items()):
        print(f'Génération de la zone {i+1}/{len(zones_affines)}')
        w = np.array(constraints[pattern]['weights'])  # Poids
        b = np.array(constraints[pattern]['biases'])   # Biais

        # Évaluation vectorisée : calculer les résultats de toutes les contraintes
        inequalities = np.dot(grid_points, w.T) + b  # (n_points x n_constraints)

        # Appliquer la normalisation Min-Max
        region_mask = ((inequalities <= 0).all(axis=1)) # Masque des points satisfaisant toutes les contraintes
        
        region_points = grid_points[region_mask]  # Filtrer les points du maillage
        c += region_points.shape[0]  # Compter les points appartenant à cette région

        if region_points.shape[0] >= 3: # Eviter les erruers dans les zones négligeable 
            hull = ConvexHull(region_points[:, :2], qhull_options="QJ")  # Crée l'enveloppe convexe si suffisant
            # Extraction des indices des points sur la bordure
            border_points = region_points[hull.vertices]


            fig.add_trace(go.Scatter3d(
                x=border_points[:, 0],
                y=border_points[:, 1],
                z=np.zeros(border_points.shape[0]),  # Fixer z=0 pour affichage 3D
                mode='lines',  # Mode pour afficher des lignes
                line=dict(
                    color="black",
                    width=3  # Épaisseur de la ligne
                ),
            ))

        else:
            print(f"Pas assez de points pour la région : {region_points.shape[0]} points disponibles.")

    

    # Ajouter le contour global du rectangle défini par x_range et y_range
    square_points = np.array([
        [x_range[0], y_range[0]],  # Coin bas gauche
        [x_range[1], y_range[0]],  # Coin bas droit
        [x_range[1], y_range[1]],  # Coin haut droit
        [x_range[0], y_range[1]],  # Coin haut gauche
        [x_range[0], y_range[0]]   # Retour au point de départ pour fermer le contour
    ])

    fig.add_trace(go.Scatter3d(
        x=square_points[:, 0],
        y=square_points[:, 1],
        z=np.zeros(square_points.shape[0]),  # Fixer z=0
        mode='lines',
        line=dict(
            color='black',  # Couleur du contour
            width=3         # Épaisseur de la ligne
        ),
        showlegend=False,
    ))

    print("Nombre de zones générées :", len(zones_affines))
    print("Nombre de points dans les zones :", c, "/", resolution**2)

    return fig.data



def plot_affine_zones_with_meshgrid_2D_border_neighbours(fig, points_pattern, x_range=(0, 1), y_range=(0, 1), resolution=100):
    border_points = []
    
    # Créer un meshgrid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # Points du maillage
    grid_shape = xx.shape  # Forme de la grille (rows, cols)

    for i in range(len(grid_points)):
        neighb = get_neighbors(i, grid_shape)

        for id_nb in neighb: 
            point = tuple(grid_points[i])  # Convertir en tuple pour utiliser comme clé
            p_nb = tuple(grid_points[id_nb])  # Voisin également en tuple

            # Vérifier si le voisin a un pattern différent
            if point in points_pattern and p_nb in points_pattern:  # Protéger contre les clés manquantes
                if points_pattern[p_nb] != points_pattern[point]:
                    border_points.append(grid_points[i])

    # Convertir border_points en numpy array pour manipulation facile
    border_points = np.array(border_points)

    if len(border_points) > 0:  # Vérifier s'il y a des points à tracer
        fig.add_trace(go.Scatter3d(
            x=border_points[:, 0],
            y=border_points[:, 1],
            z=np.zeros(border_points.shape[0]),  # Fixer z=0 pour affichage 3D
            mode='markers',
            marker=dict(
                color='black',
                size=2
            ),
            name="Border Points"
        ))

    # Ajouter le contour global du rectangle défini par x_range et y_range
    square_points = np.array([
        [x_range[0], y_range[0]],  # Coin bas gauche
        [x_range[1], y_range[0]],  # Coin bas droit
        [x_range[1], y_range[1]],  # Coin haut droit
        [x_range[0], y_range[1]],  # Coin haut gauche
        [x_range[0], y_range[0]]   # Retour au point de départ pour fermer le contour
    ])

    fig.add_trace(go.Scatter3d(
        x=square_points[:, 0],
        y=square_points[:, 1],
        z=np.zeros(square_points.shape[0]),  # Fixer z=0
        mode='lines',
        line=dict(
            color='black',  # Couleur du contour
            width=3         # Épaisseur de la ligne
        ),
        showlegend=False,
    ))
    return fig



def get_neighbors(point_index, grid_shape):

    rows, cols = grid_shape
    row, col = divmod(point_index, cols)  # Convertir l'index linéaire en indices matriciels

    neighbors = []

    # Déterminer les voisins possibles
    for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        n_row, n_col = row + d_row, col + d_col
        if 0 <= n_row < rows and 0 <= n_col < cols:  # Vérifier les limites
            neighbors.append(n_row * cols + n_col)  # Convertir les indices matriciels en index linéaire

    return neighbors



def plot_points_in_2D(X_train, fig):
    x = X_train[:, 0]
    y = X_train[:, 1]
    z = np.zeros_like(x)

    # Ajouter les points en rouge
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',  # Affiche uniquement les marqueurs
        marker=dict(color='red', size=2),  # Couleur et taille des marqueurs
        name='X_train'  # Légende
    )) 

    return fig
