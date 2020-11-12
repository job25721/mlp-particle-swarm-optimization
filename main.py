import os
import time
import csv
import numpy as np
from swarm import Model, Node
from functions import printProgressBar, cross_validation_split, select_validate
import matplotlib.pyplot as plt


def preprocessData():
    file = open('AirQualityUCI.csv', 'r')
    reader = csv.reader(file)
    data = []
    preData = {
        "3": [],
        "6": [],
        "8": [],
        "10": [],
        "11": [],
        "12": [],
        "13": [],
        "14": [],
        "desire_output": []
    }
    inputIdx = [3, 6, 8, 10, 11, 12, 13, 14]
    for i, row in enumerate(reader):
        if i > 0:
            preData['desire_output'].append(float(row[5]))
            for idx in inputIdx:
                preData[f'{idx}'].append(float(row[idx]))
    file.seek(0)
    maxOutput = np.max(preData['desire_output'])
    meanOutput = np.average(preData['desire_output'])
    inputVal = {
        "3": {"max": 0, "mean": 0},
        "6": {"max": 0, "mean": 0},
        "8": {"max": 0, "mean": 0},
        "10": {"max": 0, "mean": 0},
        "11": {"max": 0, "mean": 0},
        "12": {"max": 0, "mean": 0},
        "13": {"max": 0, "mean": 0},
        "14": {"max": 0, "mean": 0},
    }
    for idx in inputIdx:
        inputVal[f'{idx}']['max'] = np.max(preData[f'{idx}'])
        inputVal[f'{idx}']['mean'] = np.average(preData[f'{idx}'])
    for i, row in enumerate(reader):
        printProgressBar(i, 9358, 'preproceeing...', '', length=50)
        if i > 0:
            if float(row[5]) == -200:
                row[5] = meanOutput
            dic = {
                "date": row[0],
                "time": row[1],
                "input": [],
                "desire_output": float(row[5]) / maxOutput
            }

            for idx in inputIdx:
                if float(row[idx]) == -200:
                    row[idx] = inputVal[f'{idx}']['mean']
                dic['input'].append(
                    float(row[idx]) / inputVal[f'{idx}']['max'])

            data.append(dic)
    return data


def buildLayer(node):
    return [Node() for i in range(node)]


def clear(): return os.system('clear')


data = preprocessData()


def initPopulation(n):
    population = []
    print("initing...")
    for p in range(n):
        inputLayer = buildLayer(node=8)
        hiddenLayers = [buildLayer(node=4), buildLayer(node=3)]
        outputLayer = buildLayer(node=1)
        print(f'p : {p}', end=" ")
        particle = Model()
        particle.create(inputLayer, hiddenLayers, outputLayer)
        print(
            f'shape :{len(particle.inputLayer)} - {[len(l) for l in particle.hiddenLayers]} - {len(particle.outputLayer)}')
        population.append(particle)
    return population


time.sleep(3)
clear()


def velocity_vector(xpbest, xi, vi):
    return vi + (np.random.uniform(0, 1)*(xpbest - xi))


print('training..')
t_max = 25
particle_n = 10

cross_data = cross_validation_split(cross_validate_num=0.1, dataset=data)
block = cross_data["data_block"]
rand_set = cross_data["rand_set"]
reminder_set = cross_data["rem_set"]

population = initPopulation(particle_n)
first_population = population
cross_validation_plot = []
wins = []
for c in range(10):
    res = select_validate(block, rand_set, c, reminder_set)
    train = res["train"]
    cross_valid = res["cross_valid"]
    population = first_population
    printProgressBar(
        0, t_max, prefix=f'evolutioning.. t={1} ,cross_validation : {c+1}', length=25)
    tic = time.perf_counter()
    for t in range(t_max):

        printProgressBar(0, len(population), prefix='evaluating...', length=25)
        for i, p in enumerate(population):
            fx = p.evaluate(train)  # evaluate performance
            if fx < p.pbest:
                p.pbest = fx
                p.xpbest = p.x
                wins.append(p.pbest)

            # velocity calc
            v_new = velocity_vector(xpbest=p.xpbest, xi=p.x, vi=p.v)
            p.v = v_new
            # update position
            p.x = p.x + v_new
            # update weight with new position xi
            p.updateNeuralNetwork()
            printProgressBar(i+1, len(population),
                             prefix='evaluating...', suffix=f'p({i+1})fx = {fx} pbest : {p.pbest}', length=25)
        clear()
        printProgressBar(c, 10, prefix=f'overall process : cross validation : {c+1} crossValidation : {cross_validation_plot}',
                         length=50, printEnd='\n')
        np.random.shuffle(train)
        printProgressBar(
            t+1, t_max, prefix=f'evolutioning.. t={t+1}', suffix=f'best = {np.min(wins)}', length=25, printEnd='\n')
    # cross validation
    printProgressBar(0, len(population),
                     prefix='cross validationing...', length=25)
    best_validation = []
    for i, p in enumerate(population):
        mae = p.evaluate(cross_valid)
        best_validation.append(mae)
        printProgressBar(i+1, len(population),
                         prefix='cross validationing...', length=25)
    cross_validation_plot.append(np.min(best_validation))
    clear()
    toc = time.perf_counter()
    print(f'pbest in all gen = {np.min(wins)}')
    if toc-tic >= 60:
        print(f"total execution time {((toc - tic)/60):0.2f} m\n")
    else:
        print(f"total execution time {toc - tic:0.4f} secs\n")

    printProgressBar(c+1, 10, prefix=f'overall process : cross validation : {c+1} crossValidation : {cross_validation_plot}',
                     length=50, printEnd='\n')

plt.plot(cross_validation_plot)
plt.title('cross validation')
plt.show()
