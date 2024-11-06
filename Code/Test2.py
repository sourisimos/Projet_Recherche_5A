import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_voronoi_diagram(num_sites):
    # Étape 1 : Générer des sites aléatoires en 2D
    # S'assurer que les points sont éloignés des bords
    sites = np.random.rand(num_sites, 2) * 8 + 1  # Sites dans un carré de taille 10x10, mais décalés
    # Cela génère des points dans la plage [1, 9] pour éviter les bords

    # Étape 2 : Calculer le diagramme de Voronoi
    vor = Voronoi(sites)

    # Étape 3 : Afficher le diagramme de Voronoi en 2D
    fig, ax = plt.subplots(figsize=(8, 8))

    # Tracer le diagramme de Voronoi
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6)

    # Étape 4 : Afficher les sites
    ax.scatter(sites[:, 0], sites[:, 1], color='blue', label='Sites', zorder=5)  # Afficher les sites
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.title("Diagramme de Voronoi en 2D (sans régions infinies)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid()
    plt.show()

# Exécuter la fonction avec un nombre de sites souhaité
generate_voronoi_diagram(num_sites=10)  # Vous pouvez changer ce nombre
