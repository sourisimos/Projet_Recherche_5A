from maillage import MaillageDelaunayMultiDimension
from network import ReLUNetwork
import matplotlib.pyplot as plt

input_dim = 2
output_dim = 1


if __name__ == "__main__":

    # Initialistation
    mesh = MaillageDelaunayMultiDimension(10, input_dim, output_dim)
    model = ReLUNetwork(input_dim=input_dim, output_dim=output_dim, hidden_layers=10, layer_width=50)
    print(type(mesh.regions))

    X_train, Y_train = mesh.generate_points_region(10)


    # Entrainement
    model.train(X_train, Y_train, epochs=50, batch_size=32)

    point = [0.5]*input_dim
    print("Différence entre les deux réseaux pour", point,':',mesh.evaluate_function_at_point(point) - model.evaluate_point(point))


    # Affichage si 3D
    if input_dim == 2 and output_dim == 1: 
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        regions = mesh.regions

        model.find_affine_zones(regions, 100, ax)
        mesh.plot(ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Valeur de sortie (z)")
        ax.set_title("Zones Affines et Valeurs de Sortie du Réseau de Neurones ReLU en 3D")
        plt.show()
