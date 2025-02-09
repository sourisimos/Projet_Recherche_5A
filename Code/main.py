import argparse
from maillage import MaillageDelaunayMultiDimension
from network import ReLUNetwork
import plotly.graph_objects as go
from tools import plot_affine_zones_with_meshgrid_2D, plot_affine_zones_with_meshgrid_2D_border_3D, plot_affine_zones_with_meshgrid_2D_border_neighbours, plot_points_in_2D_3D, plot_affine_zones_with_meshgrid_2D_border_2D, plot_points_in_2D_2D
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
"""
Le programme peut être exécuté en ligne de commande en entrant les différentes valeurs, ou en changeant les metadatas et executant sans options. 
"""


# Metadata:
input_dim = 2 # Dimension de l'entré
output_dim = 1 # Dimension de la sorties

point_delaunay = 10 # Nb de point à partir duquel on génère la fonction affine intiale 
nb_pt_region = 100   # Nombre de point tirés par régions dans la fonction affine intiale

nb_couches_cachees = 2 # Nb de couches cachées dans le réseau
largeur_couche = 10  # Nb de neurones par couche cachée

epochs = 5050  # Nb d'époques pour l'entraînement du réseau
nb_pt_epoch = 10 # Dans le cas ou on affiche des infos sur plusieurs epochs,
grid_size = 100 # Finessse de la grille lors de l'affichage des zones affines pour le reseau 

nb_intervals = 10 # Pour train_with_intervals, donne le nombre d'interval (FIXES) considérés.

generate_cube = True # force à générer tout le cube 01 

follow_regions = False # Only display delaunay region if set to True

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
xx, yy = np.meshgrid(x, y)
grid_points = np.c_[xx.ravel(), yy.ravel()]


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

    


    """
    starts = []
    ends = []

    for i in range(1):
        print(f'Entrainement {i+1}/100')
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
        start_nb_zones, end_nb_zones = model.train_start_end_zones(X_train, Y_train, nb_epochs=epochs, batch_size=32)
        
        print(start_nb_zones)
        print(end_nb_zones)
        starts.append(start_nb_zones)
        ends.append(end_nb_zones)

    starts = np.array(starts)
    ends = np.array(ends)
    # Comparaison élément par élément
    greater = starts > ends
    # Écart à la moyenne (valeurs absolues des écarts)

    avg_dev_starts = np.mean(np.abs(starts - np.mean(starts)))
    avg_dev_ends = np.mean(np.abs(ends - np.mean(ends)))

    # Calcul de la proportion
    proportion = np.sum(greater) / len(starts)
    file_name = f"resultats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Contenu du texte
    content = f"\
    ==============Résultats==============
    Caractéristiques des réseaux
    Nb de couches: {nb_couches_cachees}
    Largeur des couches: {largeur_couche}
    Points par zones: {nb_pt_region}
    Epochs: {epochs}
    ______________________________________
    Proportion de réseaux où il y a eu diminution du nombre de zones: {proportion:.3f}
    Informations sur les zones
        Début     |Fin      
    moy:{np.mean(starts):.0f}     |{np.mean(ends):.0f}
    var:{np.var(starts):.0f}     |{np.var(ends):.0f}
    EMM:{avg_dev_starts:.0f}       |{avg_dev_ends:.0f}
    "

    # Écriture dans le fichier
    with open(file_name, "w") as file:
        file.write(content)

    print(f"Les résultats ont été enregistrés dans le fichier : {file_name}")
    """
    """
    iterate = 10
    cumul_hist_nb_zones_aff = []
    cumul_hist_val_loss = []
    cumul_hist_train_loss = []


    for i in range(iterate):
        print(f'Entrainement {i+1}/{iterate}')
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
        hist_train_loss, hist_val_loss,  hist_nb_zones_aff, hist_epoch = model.train_adapted_intervals(X_train, Y_train, nb_pt_epoch, batch_size=32)
        
        cumul_hist_nb_zones_aff.append(hist_nb_zones_aff)
        cumul_hist_train_loss.append(hist_train_loss[-1])
        cumul_hist_val_loss.append(hist_val_loss[-1])


    cumul_hist_nb_zones_aff = np.array(cumul_hist_nb_zones_aff)  # Convertir en tableau NumPy*
    mean_train_loss = np.mean(cumul_hist_train_loss)
    mean_val_loss = np.mean(cumul_hist_val_loss)


    # Calculer la moyenne et l'intervalle de confiance
    mean_curve = np.mean(cumul_hist_nb_zones_aff, axis=0)  # Moyenne sur les exécutions
    std_curve = np.std(cumul_hist_nb_zones_aff, axis=0)  # Écart-type sur les exécutions
    conf_interval = 1.96 * std_curve / np.sqrt(5)  # Intervalle de confiance 95 %

    # Tracer la courbe moyenne et l'intervalle de confiance
    # Tracer avec Plotly
    # Création du graphique
    fig = go.Figure()

    # Courbe moyenne avec intervalle de confiance
    fig.add_trace(go.Scatter(
        x=np.concatenate([hist_epoch, hist_epoch[::-1]]),  # x : concaténation pour remplir
        y=np.concatenate([mean_curve + conf_interval, (mean_curve - conf_interval)[::-1]]),  # y : supérieur + inférieur inversé
        fill='toself',  # Remplit l'intervalle entre les courbes
        fillcolor='rgba(128, 0, 128, 0.2)',  # Couleur de l'intervalle
        line=dict(width=0),  # Pas de bordure
        hoverinfo="skip",  # Pas d'info au survol
        showlegend=False  # Pas de légende
    ))

    # Ajouter la courbe moyenne
    fig.add_trace(go.Scatter(
        x=hist_epoch,
        y=mean_curve,
        mode='lines',
        name='Courbe moyenne',
        line=dict(color='blue')
    ))

    params_text = (f"Input Dimension: {input_dim}   "
                    f"Output Dimension: {output_dim}   "
                    f"Points Delaunay: {point_delaunay}<br>"
                    f"Points per Region: {nb_pt_region}   "
                    f"Hidden Layers: {nb_couches_cachees}   "
                    f"Layer Width: {largeur_couche}<br>"
                    f"Epochs: {hist_epoch[-1]}   "
                    f"Grid Size: {grid_size}   "
                    f"Generate Cube: {generate_cube}   "
                    f"Follow Regions: {follow_regions}<br>  "
                    f"Nb d'itération {iterate}   "
                    f"Train loss  {mean_train_loss}   "
                    f"Val loss  {mean_val_loss}   "


                    )

    fig.add_annotation(x=1.0,  
                        y=-0.5,
                        text=params_text,
                        showarrow=False,
                        font=dict(size=12, color="black"),
                        align="left",
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="black",
                        xref="paper",  # Référence à l'espace papier
                        yref="paper"
                        )
    
    # Mise en page
    fig.update_layout(
        title="Evolution du nombre zones affines par époques",
        xaxis_title="Époque",
        yaxis_title="Nombre de zones affines",
        template="plotly_white",
        margin=dict(l=40, r=40, t=200, b=200) 
    )

    fig.update_layout(
    xaxis=dict(
        title='Epochs',                # Titre de l'axe X
        titlefont=dict(size=30),         # Taille du titre de l'axe X
        tickfont=dict(size=26)           # Taille des ticks de l'axe X
    ),
    yaxis=dict(
        title='Zones affines',                # Titre de l'axe Y
        titlefont=dict(size=30),         # Taille du titre de l'axe Y
        tickfont=dict(size=26)           # Taille des ticks de l'axe Y
    ),
    title='Évolution du nombre de zones affines par époque',
    font=dict(size=23))  

    fig.show()
    """

    # fig.write_image(f"cosine_d{nb_couches_cachees}_w{largeur_couche}_pt{nb_pt_region}.png", width=1280, height=720)

    
    # hist_train_loss, hist_val_loss = model.train(X_train, Y_train, epochs=epochs, batch_size=64)
    # zones_affines, constraints, points_pattern = model.find_affine_zone(grid_points)


    # hist_train_loss, hist_val_loss = model.train_with_intervals(X_train, Y_train, nb_intervals, epochs, batch_size=32)

    # hist_train_loss,hist_val_loss,hist_nb_zones_aff,hist_epoch = model.train_adapted_intervals(X_train,Y_train,nb_pt_epoch)
    fig_init = go.Figure()
    mesh.plot_2D_2D(fig_init)
    plot_points_in_2D_2D(X_train, fig_init)
    model.train_adapted_intervals_v2(X_train, Y_train, fig_init)
    
    # title=f'd{nb_couches_cachees}_w{largeur_couche}'

    # model.train_adapted_intervals_animated(X_train,Y_train, fig_init, title)


    # point = [0.5] * input_dim
    # print("Différence entre les deux réseaux pour", point, ':', mesh.evaluate_function_at_point(point) - model.evaluate_point(point))


    
    # Création du tracé pour train_adapted_intervals
    

    """
    fig_plot = go.Figure()

    fig_plot.add_trace(go.Scatter(
        x=hist_epoch,                # Axe des x : époques
        y=hist_nb_zones_aff,            # Axe des y : zones affines
        mode='lines+markers',        # Ligne et marqueurs
        name='Zones Affines',        # Légende du tracé
        line=dict(width=2),           # Style de la ligne
        text=[f'Loss: {loss:.7f}' for loss in hist_train_loss],
        textposition='top center',   # Position du texte (au-dessus des points)
        textfont=dict(size=12) 
))
    params_text = (f"Input Dimension: {input_dim}   "
                    f"Output Dimension: {output_dim}   "
                    f"Points Delaunay: {point_delaunay}<br>"
                    f"Points per Region: {nb_pt_region}   "
                    f"Hidden Layers: {nb_couches_cachees}   "
                    f"Layer Width: {largeur_couche}<br>"
                    f"Epochs: {epochs}   "
                    f"Grid Size: {grid_size}   "
                    f"Generate Cube: {generate_cube}   "
                    f"Follow Regions: {follow_regions}<br>  "
                    f"Error train {hist_train_loss[-1]}   "
                    f"Val error {hist_val_loss[-1]}<br>   "
                    )

    fig_plot.add_annotation(x=1,  
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

    # Ajout des titres et des étiquettes
    fig_plot.update_layout(
        title='Évolution du nombre de zones affines par époque',
        xaxis_title='Époque',
        yaxis_title='Nombre de zones affines',
        template='plotly_white'      # Thème du graphique
    )

    # Affichage
    fig_plot.show()
    """
    
    """
    # Affichage si 3D
    
    if input_dim == 2 and output_dim == 1: 
        fig1 = go.Figure()
        fig2 = go.Figure()
        regions = mesh.regions



        # Affichage des zones en 2  D de la fonction objectif
        # mesh.plot_2D_3D(fig2)

        # Affichage des zones en 2D du rzo
        # plot_affine_zones_with_meshgrid_2D_border_3D(fig2, zones_affines, constraints)
        # plot_affine_zones_with_meshgrid_2D_border_neighbours(fig, points_pattern)


        # Affichage des points de X_train s
        # plot_points_in_2D_3D(X_train, fig2)


        
        # AFFICHAGE EN 2D 

        
        # Affichage des zones en 2  D de la fonction objectif
        mesh.plot_2D_2D(fig2)

        # Affichage des zones en 2D du rzo
        plot_affine_zones_with_meshgrid_2D_border_2D(fig2, zones_affines, constraints)
        # plot_affine_zones_with_meshgrid_2D_border_neighbours(fig2, points_pattern)


        # Affichage des points de X_train s
        plot_points_in_2D_2D(X_train, fig2)




        # Affichage de la fonciton en 3D du rzo
        model.plot_affine_zones(regions,grid_size, fig1, follow_regions)

        # Affichage de la fonction objectif en 3D 
        mesh.plot_3D(fig1)


        params_text = (f"Input Dimension: {input_dim}   "
                       f"Output Dimension: {output_dim}   "
                       f"Points Delaunay: {point_delaunay}<br>"
                       f"Points per Region: {nb_pt_region}   "
                       f"Hidden Layers: {nb_couches_cachees}   "
                       f"Layer Width: {largeur_couche}<br>"
                       f"Epochs: {epochs}   "
                       f"Grid Size: {grid_size}   "
                       f"Generate Cube: {generate_cube}   "
                       f"Follow Regions: {follow_regions}<br>  "
                       f"Error train {hist_train_loss[-1]}   "
                       f"Val error {hist_val_loss[-1]}<br>   "
                       f"Nb zones {len(zones_affines)}"
                       )

        fig1.add_annotation(x=1,  
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
        
        fig2.add_annotation(x=1,  
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
        
        
        fig1.update_layout(
            xaxis=dict(
                title='X',                # Titre de l'axe X
                titlefont=dict(size=30),         # Taille du titre de l'axe X
                tickfont=dict(size=26)           # Taille des ticks de l'axe X
            ),
            yaxis=dict(
                title='Y',                # Titre de l'axe Y
                titlefont=dict(size=30),         # Taille du titre de l'axe Y
                tickfont=dict(size=26)           # Taille des ticks de l'axe Y
            ),

            title='Fonctions objectif et générée par le réseau',
            font=dict(size=17),                   # Taille générale de la police

        )


        fig2.update_layout(
            xaxis=dict(
                title='X',                # Titre de l'axe X
                titlefont=dict(size=30),         # Taille du titre de l'axe X
                tickfont=dict(size=22)           # Taille des ticks de l'axe X
            ),
            yaxis=dict(
                title='Y',                # Titre de l'axe Y
                titlefont=dict(size=30),         # Taille du titre de l'axe Y
                tickfont=dict(size=22)           # Taille des ticks de l'axe Y
            ),


            title='Zones affines objectif et générée par le réseau',
            font=dict(size=17),                   # Taille générale de la police

        )

        fig1.show()

        fig2.show()
        """
    

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
