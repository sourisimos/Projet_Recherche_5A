import os
import re
import cv2

# Répertoire contenant vos images
image_folder = "movie/"

# Fonction pour extraire les numéros de fichier
def extract_number(filename):
    match = re.search(r'(\d+)(?=\.\w+$)', filename)  # Trouver les nombres avant l'extension
    return int(match.group()) if match else float('inf')  # Retourne un grand nombre si pas de numéro

# Charger et trier les chemins des images
image_paths = sorted(
    [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.png')],
    key=extract_number  # Utiliser la fonction pour trier
)

# Vérification
for path in image_paths[:10]:  # Affiche les 10 premiers pour vérifier
    print(path)

# Charger la première image pour obtenir la taille
frame = cv2.imread(image_paths[0])
height, width, layers = frame.shape

# Définir le codec et créer l'objet vidéo
video = cv2.VideoWriter('animation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

# Ajouter chaque image à la vidéo
for image_path in image_paths:
    frame = cv2.imread(image_path)
    video.write(frame)

# Libérer l'objet vidéo
video.release()