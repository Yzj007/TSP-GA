import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import copy

from scipy.spatial.distance import squareform, pdist

import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import copy

from scipy.spatial.distance import squareform, pdist


class TSP_GA:
    def __init__(self,
                 iteration,
                 cost_mat,
                 pop_size,
                 mutation_rate,
                 elite_rate,
                 cross_rate):
        self.iteration = iteration  # Number of iterations
        self.cost_mat = cost_mat  # Cost matrix
        self.cities = self.cost_mat.shape[0]
        self.pop_size = pop_size  # Population size
        self.mutation_rate = mutation_rate  # Mutation rate
        self.elite_rate = elite_rate  # Elite rate
        self.cross_rate = cross_rate  # Cross rate

    def initial_population(self):
        population = []
        for i in range(self.pop_size):
            cities_list = list(range(self.cities))
            random.shuffle(cities_list)
            population.append(cities_list)
        return population

    def fitness(self, individual):
        route_cost = 0
        for i in range(len(individual) - 1):
            start = individual[i]
            end = individual[i + 1]
            route_cost += self.cost_mat[start][end]
        route_cost += self.cost_mat[individual[-1]][individual[0]]
        return route_cost

    @staticmethod
    def __crossover(parent_a, parent_b):
        """
        :param parent_a: parent gene A
        :param parent_b: parent gene B
        :return:
        """
        child = [None] * len(parent_a)
        start_index, end_index = random.sample(range(len(parent_a)),
                                               2)  # Randomly generate two index points for cutting gene segments
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        child[start_index:end_index] = parent_a[start_index:end_index]
        remaining_genes = [gene for gene in parent_b if
                           gene not in child]  # Find the remaining genes in the parent gene B
        i = 0
        for gene in remaining_genes:
            while child[i] is not None:  # Find the first empty position in the child gene
                i += 1
            child[i] = gene  # Fill in the empty position with the remaining genes
        return child  # Return the child gene

    def select_elites(self, population, fitnesses):

        num_elites = int(len(population) * self.elite_rate)  # Number of elite individuals selected

        sorted_population = [individual for _, individual in sorted(
            zip(fitnesses, population))]  # Sort the population according to the fitness value of each individual

        elites = sorted_population[:num_elites]  # Select the elite individuals from the sorted population

        return elites

    @staticmethod
    def __select_two_parents(population, fitnesses):
        total_fitness = sum(fitnesses)
        selection_probability = [fitness / total_fitness for fitness in fitnesses]
        parent_a_index = random.choices(range(len(population)), weights=selection_probability, k=1)[0]
        parent_a = population[parent_a_index]
        population_without_a = population[:parent_a_index] + population[parent_a_index + 1:]
        fitnesses_without_a = fitnesses[:parent_a_index] + fitnesses[parent_a_index + 1:]
        total_fitness = sum(fitnesses_without_a)
        selection_probability = [fitness / total_fitness for fitness in fitnesses_without_a]
        parent_b_index = random.choices(range(len(population_without_a)), weights=selection_probability, k=1)[0]
        parent_b = population_without_a[parent_b_index]

        return parent_a, parent_b

    def displacement_mutation(self, individual):
        i, j = sorted(random.sample(range(len(individual)), 2))
        k = random.randint(0, len(individual) - (j - i + 1))
        genes = individual[i:j + 1]
        del individual[i:j + 1]
        individual[k:k] = genes
        return individual

    def solve(self):
        population = self.initial_population()  # init polpulation
        best_fitness = []
        for i in range(self.iteration):  # iteration
            fitnesses = [self.fitness(individual) for individual in population]  # 求解每个个体的适应度并保存为列表
            next_population = self.select_elites(population, fitnesses)  # 精英选择
            while len(next_population) < self.pop_size:
                parent_a, parent_b = self.__select_two_parents(population, fitnesses)
                if random.random() < self.cross_rate:
                    child_a = self.__crossover(parent_a, parent_b)
                    child_b = self.__crossover(parent_b, parent_a)
                else:
                    child_a = parent_a
                    child_b = parent_b
                if random.random() < self.mutation_rate:
                    child_a = self.displacement_mutation(child_a)
                    child_b = self.displacement_mutation(child_b)
                next_population.append(child_a)
                next_population.append(child_b)
            population = next_population
            fitnesses = [self.fitness(individual) for individual in population]
            best_fitness.append(min(fitnesses))
            print('iteration:{}/{},best fitness:{}'.format(i, self.iteration, min(fitnesses)))
        fitnesses = [self.fitness(individual) for individual in population]
        best_individual = population[fitnesses.index(min(fitnesses))]
        return best_individual, min(fitnesses), best_fitness
