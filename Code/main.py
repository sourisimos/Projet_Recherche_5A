import argparse
from maillage import MaillageDelaunayMultiDimension
from network import ReLUNetwork
import plotly.graph_objects as go


"""
Le programme peut être exécuté en ligne de commande en entrant les différentes valeurs, ou en changeant les metadatas et executant sans options. 
"""

# Metadata:
input_dim = 2 # Dimension de l'entré
output_dim = 1 # Dimension de la sortie

point_delaunay = 10 # Nb de point à partir duquel on génère la fonction affine intiale 
nb_pt_region = 150 # Nombre de point tirés par régions dans la fonction affine intiale

nb_couches_cachees = 10 # Nb de couches cachées dans le réseau
largeur_couche = 10 # Nb de neurones par couche cachée

epochs = 100 # Nb d'époques pour l'entraînement du réseau

grid_size = 100 #Finessse de la grille lors de l'affichage des zones affines pour le reseau 

generate_cube = True # force à générer tout le cube 01 

follow_regions = False # Only display delaunay region if set to True



def main(input_dim, output_dim, point_delaunay, nb_pt_region, nb_couches_cachees, largeur_couche, epochs, grid_size):
    # Initialistation
    mesh = MaillageDelaunayMultiDimension(point_delaunay,
                                          input_dim,
                                          output_dim,
                                          generate_cube)
    
    print("Il y a %i zones affines" % len(mesh.regions.simplices))
    model = ReLUNetwork(input_dim=input_dim,
                        output_dim=output_dim,
                        hidden_layers=nb_couches_cachees,
                        layer_width=largeur_couche)

    X_train, Y_train = mesh.generate_points_region(nb_pt_region)

    # Entrainement
    model.train(X_train, Y_train, epochs=epochs, batch_size=32)

    point = [0.5] * input_dim
    print("Différence entre les deux réseaux pour", point, ':', mesh.evaluate_function_at_point(point) - model.evaluate_point(point))


    # Affichage si 3D
    if input_dim == 2 and output_dim == 1: 
        fig = go.Figure()
        regions = mesh.regions

        model.plot_affine_zones(regions,grid_size, fig, follow_regions)
        mesh.plot(fig)

        params_text = (f"Input Dimension: {input_dim}   "
                       f"Output Dimension: {output_dim}   "
                       f"Points Delaunay: {point_delaunay}<br>"
                       f"Points per Region: {nb_pt_region}   "
                       f"Hidden Layers: {nb_couches_cachees}   "
                       f"Layer Width: {largeur_couche}<br>"
                       f"Epochs: {epochs}   "
                       f"Grid Size: {grid_size}   "
                       f"Generate Cube: {generate_cube}   "
                       f"Follow Regions: {follow_regions}"
                       )

        fig.add_annotation(x=1,  
                           y=0,
                           text=params_text,
                           showarrow=False,
                           font=dict(size=12, color="black"),
                           align="left",
                           bgcolor="rgba(255, 255, 255, 0.8)",
                           bordercolor="black",
                           xref="paper",  # Référence à l'espace papier
                           yref="paper"
                           )

        fig.update_layout(title="Zones Affines et Valeurs de Sortie du Réseau de Neurones ReLU en 3D",
                          scene=dict(xaxis_title="X",
                                     yaxis_title="Y",
                                     zaxis_title="Valeur de sortie (Z)",
                                     )   
                        )

        fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de génération de maillage et d'entraînement de réseau ReLU")
    
    # Ajout des arguments
    parser.add_argument("--input_dim", type=int, default=input_dim, help="Dimension de l'entrée")
    parser.add_argument("--output_dim", type=int, default=output_dim, help="Dimension de la sortie")
    parser.add_argument("--point_delaunay", type=int, default=point_delaunay, help="Nombre de points pour générer la fonction affine initiale")
    parser.add_argument("--generate_cube", type=bool, default=generate_cube, help="Genere sur tout le cube [0,1] ou non")
    parser.add_argument("--nb_pt_region", type=int, default=nb_pt_region, help="Nombre de points tirés par région dans la fonction affine initiale")
    parser.add_argument("--nb_couches_cachees", type=int, default=nb_couches_cachees, help="Nombre de couches cachées dans le réseau")
    parser.add_argument("--largeur_couche", type=int, default=largeur_couche, help="Nombre de neurones par couche cachée")
    parser.add_argument("--epochs", type=int, default=epochs, help="Nombre d'époques pour l'entraînement du réseau")
    parser.add_argument("--grid_size", type=int, default=grid_size, help="Finesse de la grille lors de l'affichage")
    parser.add_argument("--follow_regions", type=bool, default=follow_regions, help="Affichage uniquement de la region de Delaunay")

    # Analyse des arguments
    args = parser.parse_args()

    # Appel de la fonction principale avec les arguments
    main(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        point_delaunay=args.point_delaunay,
        nb_pt_region=args.nb_pt_region,
        nb_couches_cachees=args.nb_couches_cachees,
        largeur_couche=args.largeur_couche,
        epochs=args.epochs,
        grid_size=args.grid_size
    )
