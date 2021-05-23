#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
from random import choice, random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


# In[89]:


def preprocessor(X):
    # Preprocessing
    
    categorical_columns = [
        'job', 'marital', 'education', 'default', 
        'housing', 'loan', 'contact', 'month', 'poutcome'
    ]
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype('str'))

    return X    


# In[90]:

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = None

    def __str__(self):
        return self.chromosome

    def evaluate(self, X, y):
        if self.chromosome.count('1') == 0:
            return 0

        knn = KNeighborsClassifier(n_neighbors=3)
        columns = X.columns

        temp_columns = []
        for index, char in enumerate(self.chromosome):
            if char == '1':
                temp_columns.append(columns[index])

        temp_X = X[temp_columns]
        train_X, test_X, train_y, test_y = train_test_split(temp_X, y, test_size=0.2, random_state=0) 
        knn.fit(train_X, train_y)
        self.fitness = knn.score(test_X, test_y)
        print(self.fitness)
    
    def mutate(self, offset):
        if self.chromosome[offset] == '1':
            self.chromosome = self.chromosome[:offset] + '0' + self.chromosome[offset+1:]
        else:
            self.chromosome = self.chromosome[:offset] + '1' + self.chromosome[offset+1:]





class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, crossover_rate, mutation_rate, X, y):
        self.p = population_size
        self.l = chromosome_length
        self.c = crossover_rate
        self.m = mutation_rate
        self.X = X
        self.y = y

    def initiate_population(self):
        population = []
        for _ in range(self.p):
            chromosome = []
            for _ in range(self.l):
                chromosome.append(choice(['0', '1']))
            population.append(Individual(''.join(chromosome)))
        for index, chromosome in enumerate(population):
            print(index, chromosome)
        return population

    def evaluate_population(self, population):
        for chromosome in population:
            chromosome.evaluate(self.X, self.y)
    
    def total_fitness(self, population):
        total = 0
        for chromosome in population:
            total += chromosome.fitness
        return total

    def average_fitness(self, population):
        return self.total_fitness(population)/self.p

    def best_fitness(self, population):
        maxf = population[0].fitness
        for chromosome in population:
            if maxf < chromosome.fitness:
                maxf = chromosome.fitness
        return maxf
    
    def best_individual(self, population):
        maxf = population[0].fitness
        individual = population[0]
        for chromosome in population:
            if maxf < chromosome.fitness:
                maxf = chromosome.fitness
                individual = chromosome
        return individual

    def print_fitness(self, population):
        print('''
        Total Fitness: {}
        Average Fitness: {}
        Best Fitness: {}
        Best Individual: {}
        '''.format(
            self.total_fitness(population),
            self.average_fitness(population),
            self.best_fitness(population),
            self.best_individual(population)
        ))
    
        
    def tournament(self, population):
        temp = []
        size = len(population)
        for _ in range(size):
            parent1 = population[choice(range(size))]
            parent2 = population[choice(range(size))]

            if parent1.fitness > parent2.fitness:
                temp.append(parent1)
            else:
                temp.append(parent2)
        return temp

    def crossover(self, population):
        for i in range(0, self.p - 1, 2):
            offspring1 = []
            offspring2 = []
            xpoint = 1 + choice(range(self.l - 1))
            if random() < self.c:
                for j in range(xpoint):
                    offspring1.append(population[i].chromosome[j])
                    offspring2.append(population[i+1].chromosome[j])

                for j in range(xpoint, self.l):
                    offspring1.append(population[i+1].chromosome[j])
                    offspring2.append(population[i].chromosome[j])

            if len(offspring1) > 0 and len(offspring2) > 0:
                population[i] = Individual(''.join(offspring1))
                population[i].evaluate(self.X, self.y)
                population[i+1] = Individual(''.join(offspring2))
                population[i+1].evaluate(self.X, self.y)
        return population

    def mutate_population(self, population):
        for i in range(self.p):
            for j in range(self.l):
                if(random() < self.m):
                    population[i].mutate(j)
        return population

    def elitism(self, population):
        worst = population[0]
        worst_offset = 0
        for i in range(self.p):
            if population[i].fitness <= worst.fitness:
                worst = population[i]
                worst_offset = i
        
        best = population[0]
        best_offset = 0
        for i in range(self.p):
            if population[i].fitness >= best.fitness:
                best = population[i]
                best_offset = i
        
        population[worst_offset] = population[best_offset]
        return population

    def return_fitness(self, population):
        return {
            'Total Fitness': self.total_fitness(population),
            'Average Fitness': self.average_fitness(population),
            'Best Fitness': self.best_fitness(population),
            
        }


# In[91]:


def runner(P, C, M, G):
    df = pd.read_csv('bank.csv')
   
    X, y = df.drop('y', axis=1), df['y']

    X = preprocessor(X)

    population_size = int(P)
    chromosome_length = len(X.columns)
    crossover_rate = float(C)
    mutation_rate = float(M)
    generations = int(G)

    print(type(population_size))

    ga = GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        X=X,
        y=y
    )

    fitness_data = {}
    

    population = ga.initiate_population()
    ga.evaluate_population(population)
    ga.print_fitness(population)

    fitness_data[0] = ga.return_fitness(population)
    

    for generation in range(1, generations+1):
        
        print('Generation', generation)
        population = ga.tournament(population)
        ga.evaluate_population(population)

        population = ga.crossover(population)
        ga.evaluate_population(population)

        population = ga.mutate_population(population)
        ga.evaluate_population(population)

        population = ga.elitism(population)
        ga.evaluate_population(population)
        ga.print_fitness(population)
        
        
        fitness_data[generation] = ga.return_fitness(population)
        
    
    return fitness_data



if __name__ == '__main__':
    fitness_data = runner(10, 0.95, 0.02, 5)
    print(fitness_data)
    
    
    


# In[84]:


input_individual="0101010110110001"
L=list(input_individual)
print(L)
res=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'y']


# In[7]:


R=['0']
N=[]
for i in range(len(L)):
    if L[i] in R:
        N.append(i)
print(N)


# In[8]:


final_col=[]
for i in N:
    final_col.append(res[i])
print(final_col)
    


# In[39]:


data=pd.read_csv('bank.csv')
data.columns


# In[40]:


data.drop(columns=final_col,axis='columns',inplace=True)


# In[50]:


le = LabelEncoder()
data = data.apply(le.fit_transform)
#print(data)


# In[51]:


x = data.loc[:, data.columns!= 'y']
y = data.loc[:, data.columns == 'y']


# In[70]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 0)


# In[71]:


from sklearn import preprocessing, neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
clf = neighbors.KNeighborsClassifier()

clf.fit(x_train, y_train)

knnpre = clf.predict(x_test)
print(x_test.head())

##########Results

print(confusion_matrix(y_test,knnpre))
print(classification_report(y_test,knnpre))
KKNA = accuracy_score(y_test, knnpre)
print("The Accuracy for KNN is {}".format(KKNA))

pickle.dump(clf, open('model.pkl','wb'))
# In[ ]:




