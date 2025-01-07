import numpy as np
import tensorflow.keras as tfk # type: ignore
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools import plot_affine_zones_with_meshgrid_2D_border_2D
class ReLUNetwork:
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=10, layer_width=50):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.layer_width = layer_width
        self.model = self._create_model()

    def _create_model(self):

        # Création de l'architecture du modèle
        model = tfk.models.Sequential()

        for i in range(self.hidden_layers):
            if i == 0:
                # Spécifier input_dim uniquement pour la première couche
                model.add(
                    tfk.layers.Dense(
                        self.layer_width,
                        activation="relu",
                        input_dim=self.input_dim,
                    )
                )
            else:
                model.add(
                    tfk.layers.Dense(
                        self.layer_width,
                        activation="relu",
                    )
                )

        # Couche de sortie
        model.add(
            tfk.layers.Dense(
                self.output_dim,
            )
        )

        # Planification de l'apprentissage
        initial_lr = 1e-4
        optimizer = Adam(learning_rate=initial_lr)

        # Compilation du modèle
        model.compile(optimizer=optimizer, loss="mse")

        return model


    def train(self, X_train, y_train, epochs=50, batch_size=64):

        history = self.model.fit(
                X_train, y_train,          # Données d'entraînement
                epochs=epochs,                 # Nombre total d'époques 
                batch_size=batch_size,             # Taille du batch
                validation_split=0.2,      # Fraction des données utilisées pour la validation
                verbose=1,                  # Affiche les détails d'entraînement
                )

        return history.history['loss'], history.history['val_loss']
    


    def train_with_intervals(self, X_train, y_train, N,total_epochs=    50, batch_size=32):


        # Diviser les époques en N segments
        epochs_per_interval = total_epochs // N

        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        nb_r = N//5

        fig = make_subplots(
            rows=nb_r, cols=5,
            specs=[[{'type': 'scatter'}]*5]*nb_r,  # Type de chaque subplot
            subplot_titles=[f"Interval {i+1}" for i in range(N)],  # Titres des subplots
        )
        fig.update_layout(
            height=1000*nb_r,  # Hauteur totale (ajuster pour des subplots carrés)
            width=5000,    # Largeur totale
            showlegend=False
        )

        for interval in range(N):
            print(f"=== Début de l'intervalle {interval + 1}/{N} ===")

            # Entraîner pour le segment actuel
            history = self.model.fit(
                X_train, y_train,
                epochs=(interval+1) * epochs_per_interval,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1,
                initial_epoch=interval * epochs_per_interval,
            )

            # Obtenir les zones affines
            zones_affines, constraints, _ = self.find_affine_zone(grid_points)
            figa = go.Figure()

            # Tracer les zones affines
            fig1 = plot_affine_zones_with_meshgrid_2D_border_2D(
                figa, zones_affines, constraints, x_range=(0, 1), y_range=(0, 1), resolution=1000
            )

            # Calcul des indices de ligne et de colonne
            i = interval // 5  # Ligne
            j = interval % 5   # Colonne

            for trace in fig1:  # fig1.data contient toutes les traces
                if isinstance(trace, go.Scatter3d):  # Vérifiez si le tracé est 3D
                    # Convertir en Scatter (2D)
                    new_trace = go.Scatter(
                        x=trace.x,
                        y=trace.y,
                        mode='lines',
                        line=dict(
                            color='black',  # Couleur du contour
                            width=1         # Épaisseur de la ligne
                        )
                    )
                else:
                    new_trace = trace

                # Ajouter la trace au subplot
                fig.add_trace(new_trace, row=i+1, col=j+1)

            # Mettre à jour le titre du subplot
            fig.layout.annotations[interval].text = f"Interval {interval + 1} - Zones: {len(zones_affines)}"

        fig.show()

        return history.history['loss'], history.history['val_loss']



    def train_start_end_zones(self, X_train, y_train, nb_epochs=300, batch_size=64):
        """
        Donne le nombre de zones en début et fin d'apprentissage 
        Possibilité sur les points d'entrainement : X_train
        possibilité sur tout lespace [0,1] avec un maillage: gridpoints
        """
        history = self.model.fit(
            X_train, y_train,
            epochs=1,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            initial_epoch=0,
        )
        

        nb_zones_affines_start = self.find_number_affines_zones(X_train)

        history = self.model.fit(
            X_train, y_train,
            epochs=nb_epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            initial_epoch=1,
        )

        nb_zones_affines_end = self.find_number_affines_zones(X_train)

        return nb_zones_affines_start, nb_zones_affines_end



    def train_adapted_intervals(self, X_train, y_train, nb_pt_epoch=50, batch_size=64):
        """
        Donne le nombre de zones au cours de l'apprentissage  
        Possibilité sur les points d'entrainement : X_train
        possibilité sur tout lespace [0,1] avec un maillage: gridpoints
        """

        # Calcul des époques cumulées
        dif_epoch = np.array(range(1, nb_pt_epoch+1))  # Incréments de 1 à total_epochs
        tot_cum_epoch = np.cumsum(dif_epoch)           # Époques cumulées
        tot_epoch = tot_cum_epoch[-1]                  # Nombre total d'époques

        print(f'Nombre total d\'époques: {tot_epoch}')


        # Historique
        hist_nb_zones_aff = []
        hist_train_loss = []
        hist_val_loss =[]

        for ind in range(len(dif_epoch)):
            print(f"=== Début de l'intervalle {ind+1}/{len(dif_epoch)} ===")
            print(f"Entraînement sur {dif_epoch[ind]} époques (début à {tot_cum_epoch[ind]-dif_epoch[ind]}).")

            # Entraîner pour l'intervalle actuel
            history = self.model.fit(
                X_train, y_train,
                epochs=tot_cum_epoch[ind],  # Époque finale pour cet intervalle
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1,
                initial_epoch=tot_cum_epoch[ind] - dif_epoch[ind],  # Époque de début
            )

            # Calcul du nombre de zones affines
            nb_zones_affines = self.find_number_affines_zones(X_train)
            print(f"Nombre de zones à l'époch {tot_cum_epoch[ind]}/{tot_epoch}: {nb_zones_affines}")
            hist_nb_zones_aff.append(nb_zones_affines)

            # Enregistrer la perte
            hist_train_loss.append(history.history['loss'][-1])
            hist_val_loss.append(history.history['val_loss'][-1])


        # Générer les époques finales
        hist_epoch = tot_cum_epoch

        return (
            hist_train_loss,        # Historique des pertes d'entraînement
            hist_val_loss,          # Historique des pertes de validation
            hist_nb_zones_aff,         # Historique des zones affines
            hist_epoch              # Époques cumulées
        )
        
    def train_adapted_intervals_v2(self, X_train, y_train, fig_init, batch_size=64):
        """
        Donne le nombre de zones au cours de l'apprentissage  
        Possibilité sur les points d'entrainement : X_train
        possibilité sur tout lespace [0,1] avec un maillage: gridpoints
        """

        # Calcul des époques cumulées
        tot_cum_epoch = np.array([0]+[int((i/2)**(5)) for i in range(2,12)])        # Époques cumulées

        print(f'Nombre total d\'époques: {tot_cum_epoch[-1]}')


        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]


        fig = make_subplots(
            rows=5, cols=2,
            specs=[[{'type': 'scatter'}]*2]*5,  # Type de chaque subplot
            subplot_titles=[f"Interval {i+1}" for i in range(10)],  # Titres des subplots
        )
        fig.update_layout(
            height=1000*5,  # Hauteur totale (ajuster pour des subplots carrés)
            width=2000,    # Largeur totale
            showlegend=False
        )
        hist_train_loss =[]
        hist_val_loss = []
        for ind in range(1,len(tot_cum_epoch)):
            print(f"=== Début de l'intervalle {ind}/{len(tot_cum_epoch)-1} ===")
            print(f"Début à {tot_cum_epoch[ind-1]}")

            # Entraîner pour l'intervalle actuel
            history = self.model.fit(
                X_train, y_train,
                epochs=tot_cum_epoch[ind],  # Époque finale pour cet intervalle
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1,
                initial_epoch=tot_cum_epoch[ind-1],  # Époque de début
            )

            # Calcul du nombre de zones affines
            # Obtenir les zones affines
            zones_affines, constraints, _ = self.find_affine_zone(grid_points)
            figa = go.Figure(fig_init.to_dict())

            # Tracer les zones affines
            fig1 = plot_affine_zones_with_meshgrid_2D_border_2D(
                figa, zones_affines, constraints, x_range=(0, 1), y_range=(0, 1), resolution=1000
            )

            i = (ind - 1) // 2  # Ligne
            j = (ind - 1) % 2   # Colonne

            for trace in fig1:  # fig1.data contient toutes les traces

                # Ajouter la trace au subplot
                fig.add_trace(trace, row=i+1, col=j+1)

            # Mettre à jour le titre du subplot
            fig.layout.annotations[ind -1].text = f"Epoch {tot_cum_epoch[ind]} - Zones: {len(zones_affines)}"
            hist_train_loss.append(history.history['loss'][-1])
            hist_val_loss.append(history.history['val_loss'][-1])

        fig.show()

        return hist_train_loss, hist_val_loss


    

    def train_adapted_intervals_animated(self, X_train, y_train, fig_init, batch_size=64):
        """
        Visualise les zones affines et l'évolution de la loss au cours de l'entraînement.
        Donne également l'évolution du nombre de zones affines détectées.
        """

        # Calcul des époques cumulées
        tot_cum_epoch = np.array([0] + [int(i * 10) for i in range(0, 505)])  # Époques cumulées
        print(f"Nombre total d'époques: {tot_cum_epoch[-1]}")

        # Maillage de points pour les zones affines
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Historique des métriques
        hist_train_loss = []
        hist_val_loss = []
        hist_nb_aff_zones = []

        for ind in range(1, len(tot_cum_epoch)):
            print(f"=== Début de l'intervalle {ind}/{len(tot_cum_epoch)-1} ===")
            print(f"Début à {tot_cum_epoch[ind-1]}")

            # Entraîner pour l'intervalle actuel
            history = self.model.fit(
                X_train, y_train,
                epochs=tot_cum_epoch[ind] + 1,  # Époque finale pour cet intervalle
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1,
                initial_epoch=tot_cum_epoch[ind-1],  # Époque de début
            )

            # Vérifier et enregistrer les pertes
            if 'loss' in history.history:
                hist_train_loss.append(history.history['loss'][-1])
            else:
                print(f"Clé 'loss' manquante dans history.history à l'itération {ind}.")
                hist_train_loss.append(None)

            if 'val_loss' in history.history:
                hist_val_loss.append(history.history['val_loss'][-1])
            else:
                print(f"Clé 'val_loss' manquante dans history.history à l'itération {ind}.")
                hist_val_loss.append(None)

            # Calcul du nombre de zones affines
            zones_affines, constraints, _ = self.find_affine_zone(grid_points)
            nb_zones_affines = self.find_number_affines_zones(X_train)
            hist_nb_aff_zones.append(nb_zones_affines)            # Création des subplots
            fig_sub = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'scatter'}] * 3],
                subplot_titles=[
                    "Zones Affines",
                    "Fonction train loss",
                    "Évolution nb zones affines"
                ]
            )
            # Mise à jour des polices des titres des subplots
            for annotation in fig_sub['layout']['annotations']:
                annotation['font'] = dict(size=40)

            fig_sub.update_layout(
                width=3600,  # Largeur totale de la figure
                height=1200,  # Hauteur totale de la figure
                font=dict(size=40)  # Taille de police
            )

            # Tracer les zones affines
            figa = go.Figure(fig_init.to_dict())
            plot_affine_zones_with_meshgrid_2D_border_2D(
                figa, zones_affines, constraints, x_range=(0, 1), y_range=(0, 1), resolution=1000
            )
            for trace in figa.data:
                fig_sub.add_trace(trace, row=1, col=1)

            fig_sub.update_xaxes(title="X", row=1, col=1)
            fig_sub.update_yaxes(title="Y", row=1, col=1)

            # Tracer la loss
            fig_sub.add_trace(
                go.Scatter(x=tot_cum_epoch[:ind], y=hist_train_loss, mode='lines+markers',line=dict(width=4, color='blue'),name='Train Loss'),
                row=1, col=2
            )
            fig_sub.update_xaxes(title="Époque", range=[0, tot_cum_epoch[-1]], row=1, col=2)
            fig_sub.update_yaxes(title="Loss", range=[-5,np.log(hist_train_loss[0]*1.5)] if hist_train_loss else 1, type= 'log', row=1, col=2)

            # Tracer l'évolution du nombre de zones affines
            fig_sub.add_trace(
                go.Scatter(x=tot_cum_epoch[:ind], y=hist_nb_aff_zones, mode='lines+markers', line=dict(width=4, color='orange'), name='Nb Zones Affines',),
                row=1, col=3
            )
            fig_sub.update_xaxes(title="Époque", range=[0, tot_cum_epoch[-1]], row=1, col=3)
            fig_sub.update_yaxes(title="Nb Zones Affines (X_train)", range=[0, 70], row=1, col=3)

            # Enregistrer la figure sous forme d'image
            fig_sub.write_image(f"film_heavyside_d10_w10_pt100_{ind}.png")

        return hist_train_loss, hist_val_loss, hist_nb_aff_zones

            


    def evaluate_point(self, point):
        """
        Evalue la valeur en 1 points du réseau
        """

        # Vérifier que le point est de la bonne dimension
        point = np.array(point).reshape(1, -1)  # Reformater en (1, input_dim) pour prédiction

        # Vérifier la compatibilité de la dimension du point avec celle du réseau
        if point.shape[1] != self.input_dim:
            raise ValueError(f"La dimension du point ({point.shape[1]}) ne correspond pas à input_dim ({self.input_dim})")

        # Prédiction du modèle
        prediction = self.model.predict(point)

        return prediction[0]

    def plot_affine_zones(self, delaunay_regions, grid_size=1000, fig=None, follow_regions=False):

        """
        Affiche la fonctoin générée par le réseaux 
        """


        # Échantillonner l'espace en 2D
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)

        grid_points = np.c_[X.ravel(), Y.ravel()]

        if follow_regions:
        # Vérifier si chaque point est dans une région de Delaunay
            mask = np.array([delaunay_regions.find_simplex(point) != -1 for point in grid_points])

        # Préparer un tableau Z de même forme que X et Y, initialisé avec None pour les points hors des régions
            Z = np.full(X.shape, None, dtype=object)  # Utilisation d'un dtype object pour permettre `None`

        # Faire des prédictions uniquement pour les points valides
            valid_points = grid_points[mask]
            valid_predictions = self.model.predict(valid_points)

        # Replacer les prédictions dans Z uniquement pour les points valides
            Z[mask.reshape(X.shape)] = valid_predictions.flatten()

        else : 
            Z = self.model.predict(grid_points)
            Z = Z.reshape(grid_size, grid_size)

        # Initialiser la figure et les axes si aucun axe n'est fourni
        if fig is None:
            fig = go.Figure()

        # Afficher la surface en 3D avec les valeurs de sortie
        fig.add_trace(go.Surface(z=Z,
                                 x=X,
                                 y=Y,
                                 colorscale="Viridis",
                                 name="Réseau",
                                 opacity=1,
                                 colorbar=dict(title="Z<br>(Réseau)",
                                               len=0.6,     # Même taille que la première barre
                                               x=0.9,       # Position à droite
                                               xpad=40,     # Décalage de la barre
                                               orientation="v"  # Orientation verticale
                                               )
                                )
                    )

        return None



    def find_affine_zone(self, points):
        """
        Trouve les zones affines sur le cube 0,1 pour les points consiédéré : frontiere des régions, valeurs de la fonction affines dans les régiosn
        """


        # Capturer des activations intermédiaires
        layer_outputs = [layer.output for layer in self.model.layers if not isinstance(layer, (tfk.layers.BatchNormalization, tfk.layers.Dropout))]
        #layer_outputs = [layer.output for layer in self.model.layers]

        # Créer une fonction pour capturer les activations 
        activation_function = tfk.backend.function([self.model.input], layer_outputs)

        
        # Initialisation des dictionnaires
        zones_affines = {}
        points_pattern = {}
        constraints = {}


        # Récupération des zones d'activation par points
        for point in points:
            point = point.reshape(1, -1)  # Reformater en (1, input_dim)
            activations = activation_function([point])
            pattern = []

            # Construire le patron d'activations binaire
            for activation in activations: 
                pattern.append((activation  != 0.).astype(int).flatten()) 
            pattern = tuple(map(tuple, pattern))  # Conversion de la liste en tuple pour être la clé du dict
            if pattern not in zones_affines:
                zones_affines[pattern] = {"points": [], "matrix": None}

            zones_affines[pattern]["points"].append(point.flatten())
            points_pattern[tuple(point.flatten())] = pattern

        # Calculer les matrices affines pour chaque zone
        weights_biases = [layer.get_weights() for layer in self.model.layers if not isinstance(layer, (tfk.layers.BatchNormalization, tfk.layers.Dropout))]
        # weights_biases = [layer.get_weights() for layer in self.model.layers]
        for pattern in zones_affines.keys():
            # zone_constraints = {"weights": [], "biases": []}  # Inégalité positive ou négative
            zone_constraints = {"weights": [], "biases": []}

            W_combined = np.array(weights_biases[0][0]).T # Cracra mais je n'ai pas eu le choix ??? A résoudre proprement
            b_combined = np.array(weights_biases[0][1]).T

            W_combined_act = np.array(weights_biases[0][0] * np.array(pattern[0])).T # Cracra mais je n'ai pas eu le choix ??? A résoudre proprement
            b_combined_act = np.array(weights_biases[0][1] * np.array(pattern[0])).T

            # Premiere étape de Zone_constraint séparée 
            for neuron_idx, is_active in enumerate(np.array(pattern[0])):
                W_neuron = W_combined[neuron_idx, :]  # Poids associés au neurone
                b_neuron = b_combined[neuron_idx]    # Biais du neurone

                if is_active:  # Neurone activé : -W*x - b < 0
                    zone_constraints["weights"].append(-W_neuron)
                    zone_constraints["biases"].append(-b_neuron)

                else:          # Neurone désactivé : W*x + b <= 0
                    zone_constraints["weights"].append(W_neuron)
                    zone_constraints["biases"].append(b_neuron)
                
            for i, (u, b) in enumerate(weights_biases[1:], start= 1): # i = 0 fait hors de la boucle

                active_neurons = np.array(pattern[i])  
                # Détermination des zones de contraintes

                W_combined = np.dot(u.T, W_combined_act)
                b_combined = np.dot(u.T, b_combined_act) + b


                # Récupérer les poids et biais des neurones activés uniquement
                u_active = (u * active_neurons).T # 
                b_active = (b * active_neurons).T



                if i != len(weights_biases) -1: # la dernière couche (Linéaire, non ReLu) ne participe pas aux zones

                    for neuron_idx, is_active in enumerate(active_neurons):

                        # Récupérer les poids et biais du neurone
                        W_neuron = W_combined[neuron_idx, :] # Poids associés au neurone
                        b_neuron = b_combined[neuron_idx]    # Biais du neurone

                        # Ajouter la contrainte dans l'espace d'entrée
                        if is_active:  # Neurone activé : -W*x - b <= 0
                            zone_constraints["weights"].append(-W_neuron)
                            zone_constraints["biases"].append(-b_neuron)

                        else:          # Neurone désactivé : W*x + b <= 0
                            zone_constraints["weights"].append(W_neuron)
                            zone_constraints["biases"].append(b_neuron)
            
                # Poids utilisé pour le calcul de la fonciton affine
                W_combined_act = np.dot(u_active, W_combined_act)
                b_combined_act = np.dot(u_active, b_combined_act) + b_active


            constraints[pattern] = zone_constraints


            # Enregistrer la matrice affine et la zone de contrainte 
            zones_affines[pattern]["matrix"] = (W_combined_act, b_combined_act)



        '''        
        # Vérification de la fonction affine
        print("Vérification:")
        for _, data in (zones_affines.items()):
            for point in data['points']:
                rzo_pt = self.evaluate_point(point)
                print('rzo_pt:',rzo_pt)
                aff_pt = np.dot(data['matrix'][0], point) + data['matrix'][1] # W*x + b
                print('aff_pt',aff_pt)
                print("diff",rzo_pt-aff_pt)
        '''

        """
        # Vérification des zones affines
        print('Vérification')
        for pattern, data in (zones_affines.items()):
            for point in data['points']:
                print('-------------')
                print("POS")
                for w, b in zip(constraints[pattern]['weights'], constraints[pattern]['biases']): 
                    bol = (np.dot(w, point) + b >= 0)
                    print(bol)
                    if not bol : 
                        print(np.dot(w, point) + b)
        """ 

        return zones_affines, constraints, points_pattern
    


    def find_number_affines_zones(self, points):
        """
        Calcul le nombre de zones affines pour les points considérés. 
        """

        # Capturer des activations intermédiaires
        layer_outputs = [layer.output for layer in self.model.layers if not isinstance(layer, (tfk.layers.BatchNormalization, tfk.layers.Dropout))]
        #layer_outputs = [layer.output for layer in self.model.layers]

        # Créer une fonction pour capturer les activations 
        activation_function = tfk.backend.function([self.model.input], layer_outputs)

        
        # Initialisation des dictionnaires
        zones_affines = {}

        # Récupération des zones d'activation par points
        for point in points:
            point = point.reshape(1, -1)  # Reformater en (1, input_dim)
            activations = activation_function([point])
            pattern = []

            # Construire le patron d'activations binaire
            for activation in activations: 
                pattern.append((activation  != 0.).astype(int).flatten()) 
            pattern = tuple(map(tuple, pattern))  # Conversion de la liste en tuple pour être la clé du dict
            if pattern not in zones_affines:
                zones_affines[pattern] = {"points": []}
            zones_affines[pattern]["points"].append(point.flatten())
        
        nb_zones = len(zones_affines)
        return nb_zones