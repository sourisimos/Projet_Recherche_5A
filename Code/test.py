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
        

    def assurer_continuite(self, points_per_frontier=5):
        # Pour chaque paire de régions adjacentes, ajuster les applications affines
        for i in range(self.nb_regions):
            for j in range(i + 1, self.nb_regions):
                A_i, b_i = self.regions[i]
                A_j, b_j = self.regions[j]
                
                # Trouver plusieurs points à la frontière commune entre les régions i et j
                x_f_list = self.trouver_points_a_la_frontiere(A_i, b_i, A_j, b_j, points_per_frontier)
                if len(x_f_list) > 0:
                    M_i, c_i = self.applications[i]
                    M_j, c_j = self.applications[j]
                    
                    # Ajuster les matrices et les vecteurs constants en minimisant les écarts sur plusieurs points
                    delta_M, delta_c = self.resoudre_continuite_multi_points(M_i, c_i, M_j, c_j, x_f_list)
                    
                    # Appliquer les ajustements à M_i et M_j
                    self.applications[i] = (M_i + delta_M / 2, c_i - delta_c / 2)
                    self.applications[j] = (M_j - delta_M / 2, c_j + delta_c / 2)


    def resoudre_continuite_multi_points(self, M_i, c_i, M_j, c_j, x_f_list):
        # On veut résoudre l'égalité M_i @ x_f + c_i = M_j @ x_f + c_j pour plusieurs points x_f
        # Formons un système surdéterminé (moindres carrés)

        # Créer des matrices pour le système d'équations linéaires
        A = []
        B = []

        for x_f in x_f_list:
            diff_M = M_i - M_j
            diff_c = c_i - c_j
            
            A.append(np.hstack([x_f[:, np.newaxis], np.ones((x_f.shape[0], 1))]))
            B.append(np.hstack([diff_M @ x_f, diff_c]))

        A = np.vstack(A)  # Combiner tous les points en une seule matrice
        B = np.vstack(B)  # Vecteur de différences

        # Résolution par moindres carrés
        solution, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        
        delta_M = solution[:-1]  # Ajustement à apporter à M
        delta_c = solution[-1]   # Ajustement à apporter à c
        
        return delta_M, delta_c

    def trouver_points_a_la_frontiere(self, A_i, b_i, A_j, b_j, points_per_frontier):
        # Trouver plusieurs points satisfaisant les deux inégalités (A_i @ x <= b_i) et (A_j @ x <= b_j)
        # On peut générer des points aléatoires ou utiliser une approche basée sur la programmation linéaire.
        from scipy.optimize import linprog
        
        points = []
        for _ in range(points_per_frontier):
            c = np.random.randn(self.dim_entree)  # Générer une direction aléatoire
            A_ineq = np.vstack([A_i, A_j])
            b_ineq = np.hstack([b_i, b_j])
            
            res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, method='highs')
            
            if res.success:
                points.append(res.x)
        
        return points

    def evaluer(self, x):
        self.assurer_continuite()
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

# Exemple d'utilisation
dim_entree = 2
dim_sortie = 1
nb_regions = 4

fonction_pwl = FonctionAffineParMorceaux(dim_entree, dim_sortie, nb_regions)

for i in range(nb_regions):
    rang = np.random.randint(1, min(dim_entree, dim_sortie) + 1)  # Rang aléatoire pour chaque région
    fonction_pwl.ajouter_region(rang)

x_test = np.random.randn(dim_entree)
y_test = fonction_pwl.evaluer(x_test)
print(y_test)
