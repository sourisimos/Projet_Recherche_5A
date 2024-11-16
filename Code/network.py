import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import plotly.graph_objects as go

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
        crée un réseau de neurones avec des couches ReLU.
        retourne : le model en configuration initial (objet TF)
        """

        model = Sequential()
    
        for i in range(self.hidden_layers):
            if i == 0:
                # Spécifier input_dim uniquement pour la première couche
                model.add(tfk.layers.Dense(self.layer_width, activation='relu', input_dim=self.input_dim))
            else:
                model.add(tfk.layers.Dense(self.layer_width, activation='relu'))

        model.add(Dense(self.output_dim))  # Sortie sans activation pour valeurs continues

        model.compile(optimizer='adam', loss='mse')

        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Entraîne le modèle sur les données fournies.

        X_train : Données d'entrée d'entraînement.
        y_train : Valeurs cibles d'entraînement.
        epochs : Nombre d'époques d'entraînement.
        batch_size : Taille de lot pour l'entraînement.

        retourne : NOne
        """
    
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        return None

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

    def find_affine_zones(self, delaunay_regions, grid_size=50, fig=None, follow_regions=False):
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

