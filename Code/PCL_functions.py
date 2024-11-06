import numpy as np

class FonctionAffineParMorceaux:
    def __init__(self, dim_entree, dim_sortie, nb_regions):
        self.dim_entree = dim_entree
        self.dim_sortie = dim_sortie
        self.nb_regions = nb_regions
        self.regions = []
        self.applications = []
    
    def generer_region_polyedrique(self):
        # Générer une région polyédrique à l'aide d'inégalités aléatoires
        A = np.random.randn(self.dim_entree, self.dim_entree)
        b = np.random.randn(self.dim_entree)
        return A, b
    
    def generer_application_affine(self, rang):
        # Générer une application affine aléatoire avec un rang donné
        M = np.random.randn(self.dim_sortie, self.dim_entree)
        # Réduire le rang en mettant certaines valeurs singulières à zéro si nécessaire
        if rang < min(self.dim_sortie, self.dim_entree):
            u, s, vh = np.linalg.svd(M, full_matrices=False)
            s[rang:] = 0  # Mettre à zéro les valeurs singulières inférieures
            M = u @ np.diag(s) @ vh
        c = np.random.randn(self.dim_sortie)
        return M, c
    
    def ajouter_region(self, rang):
        A, b = self.generer_region_polyedrique()
        M, c = self.generer_application_affine(rang)
        self.regions.append((A, b))
        self.applications.append((M, c))

    def evaluer(self, x):
        print(f"Évaluation de l'entrée: {x}")
        for idx, ((A, b), (M, c)) in enumerate(zip(self.regions, self.applications)):
            # Affiche les valeurs de A, b pour chaque région et le test logique
            print(f"Région {idx}: A = {A}, b = {b}")
            print(f"Résultat de A @ x <= b : {A @ x} <= {b} => {A @ x <= b}")
            if np.all(A @ x <= b):
                print(f"L'entrée x appartient à la région {idx}")
                return M @ x + c
        # Si aucune région ne contient l'entrée
        raise ValueError("L'entrée x n'appartient à aucune région")


    
    def assurer_continuite(self):
        # Assurer la continuité entre les régions adjacentes en modifiant les applications affines
        # Cette étape implique de résoudre des équations linéaires pour faire correspondre les conditions aux frontières
        pass

# Exemple d'utilisation
dim_entree = 2
dim_sortie = 3
nb_regions = 4

fonction_pwl = FonctionAffineParMorceaux(dim_entree, dim_sortie, nb_regions)

for i in range(nb_regions):
    rang = np.random.randint(1, min(dim_entree, dim_sortie) + 1)  # Rang aléatoire pour chaque région
    fonction_pwl.ajouter_region(rang)

x_test = np.random.randn(dim_entree)
y_test = fonction_pwl.evaluer(x_test)
print(y_test)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fonction pour tracer un plan 3D défini par une équation ax + by + cz = d
def plot_plane(ax, A, b, xlim, ylim, color='b', alpha=0.5):
    # On suppose que l'équation est de la forme Ax + By + Cz <= D
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
    if A[2] != 0:  # Eviter la division par 0
        zz = (b - A[0] * xx - A[1] * yy) / A[2]
        ax.plot_surface(xx, yy, zz, color=color, alpha=alpha)

# Visualiser les régions en 3D et les points évalués
def visualiser_regions_3D(pwl_function, x_min, x_max, num_points=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Générer une grille de points en 3D
    x_values = np.linspace(x_min, x_max, num_points)
    y_values = np.linspace(x_min, x_max, num_points)
    z_values = np.linspace(x_min, x_max, num_points)
    X, Y, Z = np.meshgrid(x_values, y_values, z_values)

    # Afficher les points et leur région d'appartenance
    for i in range(num_points):
        for j in range(num_points):
            for k in range(num_points):
                point = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                try:
                    # Évaluer la fonction PWL
                    result = pwl_function.evaluer(point)
                    # Colorer les points en fonction de leur région
                    ax.scatter(point[0], point[1], point[2], color='g')
                except ValueError:
                    # Points hors des régions (catch-all ou non-appartenance)
                    ax.scatter(point[0], point[1], point[2], color='r')

    # Tracer les frontières des régions polyédriques (les plans)
    for idx, (A, b) in enumerate(pwl_function.regions):
        for i in range(A.shape[0]):  # Chaque inégalité définit un plan
            plot_plane(ax, A[i, :], b[i], (x_min, x_max), (x_min, x_max), color='b', alpha=0.3)

    # Ajuster les limites des axes
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([x_min, x_max])
    ax.set_zlim([x_min, x_max])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Appel de la fonction pour visualiser
visualiser_regions_3D(fonction_pwl, -5, 5)
