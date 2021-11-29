import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# Hyperparamètres

OF = 7
MAX_GEN = 5000

# models représente la liste des réseaux
# data est composé des entrées et sorties de la série temporelle
def fitness_function(models, data) :
    mse_values = []
    x = data[0]
    y = data[1]
    for network in models :
        predictions = network.predict(x)
        
        mse = mean_squared_error(y, predictions)
        mse_values.append(mse)

    return np.asarray(mse_values)

def scale_fitness(x) :
    return 1 / (1 + x)

#----------------------------------------------------------------------------------------------#
#                                                                                              #
#                                       Neural Network                                         #
#                                                                                              #
#----------------------------------------------------------------------------------------------#

# Fonction d'Activation Unité Linéaire Rectifiée (ReLU - Rectified Linear Unit)

def relu(x):
    return np.maximum(x, 0)

# Un Réseau de Neurones utilisé pour l'Evolution
# L'algorithme créera une liste de ces objets

class EvolvableNetwork:

    # Layer Nodes est une liste de valeur entières désignant le nombre de noeuds par couches
    # Par exemple, si layer_nodes = [3, 5, 3], 
    # on a 3 couches cachées avec respectivement 3, 5 et 3 noeuds
    # num_input et num_output font référence au nombre de variables d'entrées et de sorties
    # Si le booléen Initialize est a False, on ne créé aucun poids ou matrice de biais. 
    # Tous les poids et matrices de biais sont initialisées uniformément de -1 a 1

    def __init__(self, layer_nodes, num_input, num_output, initialize = True) :
        self.layer_count = len(layer_nodes)
        self.layer_nodes = layer_nodes
        self.num_input = num_input
        self.num_output = num_output
        self.activation_function = relu # Pointeur sur fonction

        self.layer_weights = []
        self.layer_biases = []

        if not initialize :
            return

        # Créé une matrice NxM pour les poids et une matrice de biais pour la couche d'entrée
        self.layer_weights.append(np.random.uniform(-1, 1, num_input * layer_nodes[0]).reshape(num_input, layer_nodes[0]))
        self.layer_biases.append(np.random.uniform(-1, 1, layer_nodes[0]))

        # Créé les poids et biais pour les couches cachées
        for i in range(1, self.layer_count) :
            self.layer_weights.append(np.random.uniform(-1, 1, layer_nodes[i-1]*layer_nodes[i]).reshape(layer_nodes[i-1], layer_nodes[i]))
            self.layer_biases.append(np.random.uniform(-1, 1, layer_nodes[i]).reshape(1, layer_nodes[i]))
        
        # Création des poids et biais pour la couche de sortie
        self.layer_weights.append(np.random.uniform(-1, 1, layer_nodes[self.layer_count-1]*num_output).reshape(layer_nodes[self.layer_count-1], num_output))
        self.layer_biases.append(np.random.uniform(-1, 1, num_output).reshape(1, num_output))

    # Pareil que forward pass, applique la multiplication des matrices de poids
    def predict(self, x) :
        output = self.activation_function(np.dot(x, self.layer_weights[0]) + self.layer_biases[0])
        for i in range(1, self.layer_count + 1) :
            if i == self.layer_count : # Dernière couche, donc pas de fonction d'activation
                output = (np.dot(output, self.layer_weights[i]) + self.layer_biases[i])
            else :
                output = self.activation_function(np.dot(output, self.layer_weights[i]) + self.layer_biases[i])

        if self.num_output == 1 and not(len(output) == 1): # Si il n'y a qu'une seule variable de sortie, alors reshape
            return output.reshape(len(x), )

        return output

#----------------------------------------------------------------------------------------------#
#                                                                                              #
#                                       Genetic Algorithm                                      #
#                                                                                              #
#----------------------------------------------------------------------------------------------#

def roulette_wheel_selection(cumulative_sum, n) :
    ind = []
    r = np.random.uniform(0, 1, n)
    for i in range(0, n) :
        index = 0
        while cumulative_sum[index] < r[i] :
            index += 1
        ind.append(index)
    return ind

# p1 et p2 sont les parents
# const_cross est le coefficient pour la combinaison linéaire, compris entre [0, 1]
# Si c'est proche de 0, cela favorisera p1, si proche de 1, cela favorisera p1,
# si c'est égal à 0.5, ce sera la moyenne de p1 et p2

def crossover(p1, p2, const_cross) :
    # Initialise le nouveau réseau avec des couches et noeuds vides
    child = EvolvableNetwork(layer_nodes = p1.layer_nodes, num_input = p1.num_input, num_output = p1.num_output, initialize = False)

    # Rempli les matrices de poids et biais des enfant avec celle des parents
    for i in range(0, p1.layer_count+1) :
        child.layer_weights.append((1 - const_cross)*p1.layer_weights[i]+const_cross*p2.layer_weights[i])
        child.layer_biases.append((1 - const_cross)*p1.layer_biases[i]+const_cross*p2.layer_biases[i])

    return child

# const_mutate est la valeur maximum de mutation
def mutation(child, const_mutate) :
    # Itérer sur toutes les couches
    for i in range(0, child.layer_count+1) :
        n, c = child.layer_weights[i].shape
        # Poids aléatoires à ajouter à l'enfant courant
        r_w = np.random.uniform(-const_mutate, const_mutate, n*c)
        # Itérer sur toutes les lignes et colonnes de la couche actuelle
        for nr in range(0, n) :
            for nc in range(0, c) :
                child.layer_weights[i][nr, nc] += r_w[nr*c+nc]

    # Itérer sur toutes les couches
    for i in range(0, child.layer_count+1) :
        c = child.layer_biases[i].shape[0]
        # Poids aléatoires à ajouter à l'enfant courant
        r_w = np.random.uniform(-const_mutate, const_mutate, c)
        # Itérer sur toutes les colonnes du vecteur
        for nc in range(0, c) :
            child.layer_biases[i][nc] += r_w[nc]

# p1 et p2 sont les parents
# cross_mutate est la valeur maximale de mutation

def reproduce(p1, p2, const_mutate, train_data) :

    # Créé un coefficient gamma différent pour normalisation
    # Croisement pour chaque enfant
    c_cross = np.random.normal(0.5, 0.15, 4)
    ch1 = crossover(p1, p2, c_cross[0])
    ch2 = crossover(p1, p2, c_cross[1])
    ch3 = crossover(p1, p2, c_cross[2])
    ch4 = crossover(p1, p2, c_cross[3])

    # Muter uniquement deux individus
    mutation(ch3, const_mutate)
    mutation(ch4, const_mutate)

    # Groupement des enfants avec les parents
    all = [p1, p2, ch1, ch2, ch3, ch4]
    fit = fitness_function(all, train_data)

    # Retourne l'individu avec le plus faible fitness
    return all[np.argmin(fit)]

# const_mutate est la valeur maximum de mutation
# Utilise train et val data pour prévenir le surentrainement
# Arrêt prématuré si la moyenne de val_data augmente trois fois d'affilés

def evolve(init_gen, const_mutate, max_iter, train_data, val_data) :
    gen = init_gen
    mean_fitness = []
    val_mean = [] # Moyenne de validation
    best_fitness = []
    prev_val = 1000
    n = len(gen)
    val_index = 0
    for k in range(0, max_iter) :
        fitness = fitness_function(gen, train_data)
        # Mise à l'échelle inverse
        scaled_fit = scale_fitness(fitness)

        # Création de la distribution pour la sélection proportionelle
        fit_sum = np.sum(scaled_fit)
        fit = scaled_fit / fit_sum
        cumulative_sum = np.cumsum(fit)

        selected = roulette_wheel_selection(cumulative_sum, n)
        mates = roulette_wheel_selection(cumulative_sum, n)

        children = []
        for i in range(0, n) :
            children.append(reproduce(gen[selected[i]], gen[mates[i]], const_mutate, train_data))

        gen_next = children

        # Evaluation des données d'entrainement
        fit = fitness_function(gen_next, train_data)
        fit_mean = np.mean(fit)
        fit_best = np.min(fit)
        mean_fitness.append(fit_mean)
        best_fitness.append(fit_best)

        # Evaluation des données de validation
        val_fit = fitness_function(gen_next, val_data)
        val_fit_mean = np.mean(val_fit)
        val_mean.append(val_fit_mean)

        print(f"Génération: {str(k)}\n Meilleur: {fit_best}, Moyenne: {fit_mean}\n Validation: {val_fit_mean}")

        gen = gen_next
        # Vérifie si la dernière itération s'est améliorée ou détériorée
        if val_fit_mean > prev_val :
            val_index += 1
        else :
            val_index = 0
        if val_index == OF : # val a augmentée pendant les trois dernières itérations
            print("Over Fitting, Arrêt...")
            break
        prev_val = val_fit_mean

    # Utilisation des données de validation pour choisir le meilleur modèle de la génération
    val_fit = fitness_function(gen_next, val_data)
    best_val = np.min(val_fit)
    best_ind = np.argmin(val_fit)
    print(f"Meilleur modèle:\n Validation: {best_val}")
    return gen_next[best_ind]

#----------------------------------------------------------------------------------------------#
#                                                                                              #
#                                Dataset Formating & Training                                  #
#                                                                                              #
#----------------------------------------------------------------------------------------------#

df = pd.read_csv("Covid_France.csv")
y = np.asarray(df['Nouveaux_Cas_lisse'])
size = len(y)
# 50% des données pour l'entraînement
train_ind = int(size*0.50)
# 25% des données pour validation et les derniers 25% pour les tests
val_ind = int(size * 0.75)

max_window = 10
min_window = 3
initial_population_size = 100 # 10 réseaux de neurones
best_models = [] # Meilleur modèle de chaque génération par taille de fenêtre
best_fits = []
# Mélange les données aléatoirement par index
shuffled_indices = np.asarray(range(0, size-max_window))
np.random.shuffle(shuffled_indices)
# Itère pour chaque taille de fenêtre
for vision in range(min_window, max_window + 1) :
    input = []
    output = []
    # Créé la taille de fenêtre pour chaque valeur, on ignore le premier couple puisqu'il n'y aura pas de fenêtre complète
    # C'est pourquoi on commence a i et non 0
    for j in range(vision, size) :
        input.append(y[(j - vision):j].tolist())
        output.append(y[j])

    input = np.asarray(input)
    output = np.asarray(output)

    temp = np.column_stack((output, input))

    # Au lieu de mélanger a chaque fois, on mélange une fois en dehors de la boucle, pour que toutes les fenêtres ait le même array final
    temp = temp[shuffled_indices]

    output = temp[:, 0]
    input = temp[:, 1:]

    y_train = output[0:train_ind]
    y_val = output[train_ind:val_ind]
    y_test = output[val_ind:size]
    x_train = input[0:train_ind]
    x_val = input[train_ind:val_ind]
    x_test = input[val_ind:size]

    init_gen = []
    for i in range(0, initial_population_size) :
        init_gen.append(EvolvableNetwork(layer_nodes = [5, 5, 5], num_input = vision, num_output = 1, initialize = True))

    best_model = evolve(init_gen, const_mutate = 0.1, max_iter = MAX_GEN, train_data = [x_train, y_train], val_data = [x_val, y_val])
    best_models.append(best_model)
    best_fits.append(fitness_function([best_model], [x_val, y_val]))

# Récupération du meilleur modèle
best_index = np.argmin(best_fits)
best_model = best_models[best_index]

# Recréation des données avec cette valeur de taille de fenêtre
vision = best_index + min_window
input = []
output = []
for j in range(vision, size) :
    input.append(y[(j - vision):j].tolist())
    output.append(y[j])

input = np.asarray(input)
output = np.asarray(output)
temp = np.column_stack((output, input))
temp = temp[shuffled_indices]
output_2 = temp[:, 0]
input_2 = temp[:, 1:]
y_test = output_2[val_ind:size]
x_test = input_2[val_ind:size]

# Evaluation sur les données de test
mse_test = fitness_function([best_model], [x_test, y_test])
print("\nMeilleur fitness de Validation par taille de fenêtre:")
index = 0
for fit in best_fits :
    print(f"Taille de fenêtre: {index+min_window} - MSE Validation: {best_fits[index][0]}")
    index += 1
print(f"Meilleur modèle:\n Taille de fenêtre: {best_index+3}\n MSE sur les données de Test: {mse_test[0]}")

# Extrapolation
extrapolation = 24

for i in range(0, extrapolation) :
    sel = np.asarray([input[len(input)-1, 1:].tolist()])
    res = best_model.predict(input[len(input)-1])
    tmp = np.append(sel, res, axis = 1)
    input = np.append(input, tmp, axis = 0)
    y = np.append(y, np.asarray([np.nan]), axis=0)

# Saving Model
mse_file = open("mse_file.tab", "ab")
models_file = open("models.brain", "ab")

pickle.dump(mse_test[0], mse_file)
pickle.dump(best_model, models_file)

mse_file.close()
models_file.close()

# Création du graphique final
xaxis = range(vision, len(y))
plt.plot(xaxis, y[vision:], label="Actuel")
plt.plot(xaxis, best_model.predict(input), label = "Prédiction")
plt.xlabel("Jour")
plt.ylabel("Nombre de cas")
plt.title("Evolution des cas covid en france")
plt.legend()
plt.show()
