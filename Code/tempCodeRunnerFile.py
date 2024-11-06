        
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
    