import numpy as np
import tensorflow.keras as tfk # type: ignore
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal, RandomNormal
from tensorflow.keras.callbacks import ReduceLROnPlateau
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools import plot_affine_zones_with_meshgrid_2D_border
class ReLUNetwork:
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=10, layer_width=50):
        """
        Initialise un réseau ReLU pour l'approximation de fonction.

        input_dim : Dimension d'entrée du réseau (par exemple, 2 pour un pavage 2D).
        output_dim : Dimension de sortie du réseau (par exemple, 1 pour des valeurs scalaires).
        hidden_layers : Nombre de couches cachées.
        layer_width : Nombre de neurones par couche cachée.
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.layer_width = layer_width
        self.model = self._create_model()

    def _create_model(self):
        """
        Crée un réseau de neurones avec des couches ReLU.
        retourne : le modèle initialisé (objet TF).
        """

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
                        kernel_initializer=HeNormal(),
                    )
                )
            else:
                model.add(
                    tfk.layers.Dense(
                        self.layer_width,
                        activation="relu",
                        kernel_initializer=HeNormal(),
                    )
                )

        # model.add(tfk.layers.BatchNormalization())
        # model.add(tfk.layers.Dropout(0.3))  # Dropout 30%

        # Couche de sortie
        model.add(
            tfk.layers.Dense(
                self.output_dim,
                kernel_initializer=RandomNormal(
                    mean=0.0, stddev=np.sqrt(1.0 / self.layer_width)
                ),
            )
        )

        # Planification de l'apprentissage
        initial_lr = 0.01
        optimizer = Adam(learning_rate=initial_lr)

        # Compilation du modèle
        model.compile(optimizer=optimizer, loss="mse")

        return model


    def train(self, X_train, y_train, epochs=50, batch_size=16):
        """
        Entraîne le modèle sur les données fournies.

        X_train : Données d'entrée d'entraînement.
        y_train : Valeurs cibles d'entraînement.
        epochs : Nombre d'époques d'entraînement.
        batch_size : Taille de lot pour l'entraînement.

        retourne : NOne
        """
        # self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        history = self.model.fit(
                X_train, y_train,          # Données d'entraînement
                epochs=epochs,                 # Nombre total d'époques
                batch_size=batch_size,             # Taille du batch
                validation_split=0.2,      # Fraction des données utilisées pour la validation
                verbose=1,                  # Affiche les détails d'entraînement
                )



        return history.history['loss'][-1], history.history['val_loss'][-1]
    
    def train_with_intervals(self, X_train, y_train, total_epochs=50, batch_size=32):
        """
        Entraîne le modèle par intervalles et affiche les zones affines après chaque segment.

        Args:
        - X_train : Données d'entrée d'entraînement.
        - y_train : Valeurs cibles d'entraînement.
        - total_epochs : Nombre total d'époques.
        - batch_size : Taille des lots.
        - grid_size : Taille de la grille pour afficher les zones affines.
        - delaunay_regions : Optionnel, pour filtrer avec des régions Delaunay.
        """

        reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # Variable surveillée
        factor=0.50,          # Facteur de réduction (par ex. divise par 2)
        patience=5,          # Nombre d'époques sans amélioration avant réduction
        min_lr=1e-6,         # Taux d'apprentissage minimum
        verbose=1            # Affiche les changements
    )
        # Diviser les époques en 10 segments
        epochs_per_interval = total_epochs // 10

        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        fig = make_subplots(
            rows=2, cols=5,
            specs=[[{'type': 'scatter'}]*5]*2,  # Type de chaque subplot
            subplot_titles=[f"Interval {i+1}" for i in range(50)],  # Titres des subplots
        )
        fig.update_layout(
            height=5000,  # Hauteur totale (ajuster pour des subplots carrés)
            width=2000,    # Largeur totale
            showlegend=False
        )

        for interval in range(10):

            print(f"=== Début de l'intervalle {interval + 1}/50 ===")

            # Entraîner pour le segment actuel
            history = self.model.fit(
                X_train, y_train,
                epochs=(interval+1) * epochs_per_interval,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1,
                initial_epoch=interval * epochs_per_interval,  # Reprend là où on s'est arrêté
                callbacks = [reduce_lr]
            )

            # Get inequalities 
            zones_affines, constraints, _ = self.find_affine_zone(grid_points)
            figa = go.Figure()
            # plot affines zones
            fig1 = plot_affine_zones_with_meshgrid_2D_border(figa, zones_affines, constraints, x_range=(0, 1), y_range=(0, 1), resolution=1000)

            i = interval%5
            j = interval//5
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

                fig.add_trace(new_trace, row=i+1, col=j+1)

                fig.layout.annotations[interval].text = f"Interval {interval + 1} - Zones: {len(zones_affines)}"

        fig.show()
        print("=== Entraînement terminé ===")
        return history.history['loss'][-1], history.history['val_loss'][-1]


    def evaluate_point(self, point):
        """
        Évalue la valeur prédite par le réseau pour un point donné.

        point : np.array ou liste de taille (input_dim,) - coordonnées du point à évaluer.

        Retourne : La valeur prédite pour le point.
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
        Génère les zones affines apprises par le réseaux à partir d'un grillage

        delaunay_regions : Coordonnées des régions de base (scipy.spatial._qhull.Delaunay)
        grid_size : subdivision de la grille 
        ax : valeur d'ax (type :'mpl_toolkits.mplot3d.axes3d.Axes3D')

        retourne: None
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

        # print(constraints)
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