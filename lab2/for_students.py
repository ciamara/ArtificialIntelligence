from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from data import *

#crates population(random items from data) of size population_size
#each individual true/false, either chosen or not
#individual_size -> amount of individuals
def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

#checks ,,value'' of knapsack based on true/false list
def fitness(items, knapsack_max_capacity, individual):
    #checks if weight is valid
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0    # too heavy
    return sum(compress(items['Value'], individual))    #accepts knapsack -> returns value of whole knapsack

# finds the best individual based on value
def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness # returns the best individual

# roulette selection
def roulette_selection(items, knapsack_max_capacity, population, n_selection):
    fitness_values = [fitness(items, knapsack_max_capacity, ind) for ind in population]
    total_fitness = sum(fitness_values)

    if total_fitness == 0:
        return random.choices(population, k=n_selection)  # if all fitnesses =0, choose randomly

    selection_probabilities = [f / total_fitness for f in fitness_values]  # probabilities
    selected = np.random.choice(len(population), size=n_selection, replace=True, p=selection_probabilities)# choosing based on probabilities
    
    return [population[i] for i in selected]    # returning selected

# (krzyzowanie jednopunktowe), child1 -> 1/2parent1 + 1/2parent2, child2 -> unused halves for previous child
def single_point_crossover(parent1, parent2):
    length = len(parent1)
    crossover_point = length // 2  # halves

    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2

# crossover (krzyzowanie)
def crossover(population):
    new_population = []
    random.shuffle(population)  # shuffle for random parent choices

    # creating new population using single point crossover
    for i in range(0, len(population) - 1, 2):
        parent1, parent2 = population[i], population[i + 1] #choosing parents
        child1, child2 = single_point_crossover(parent1, parent2) # creating children using single point crossover
        new_population.extend([child1, child2]) # adding children to new population

    return new_population

# one bit flip per individual
def mutate(individual):
    mutation_point = random.randint(0, len(individual) - 1)  # Losowy indeks
    individual[mutation_point] = not individual[mutation_point]  # Negacja wartości
    return individual

# mutating population using bit flip(one flip per individual)
def mutate_population(population):
    return [mutate(individual) for individual in population]

# get data
items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)

for _ in range(generations):
    population_history.append(population)   #current population

    # TODO: implement genetic algorithm-----------------------------------------------------------

    # roulette selection of parents
    population = roulette_selection(items, knapsack_max_capacity, population, n_selection)

    #crossover -> new population using single point crossover
    population = crossover(population)

    # mutation(bit flip)
    population = mutate_population(population)

    #----------------------------------------------------------------------------------------------

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)    #best individual
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)   #saves best fitness from generation

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
