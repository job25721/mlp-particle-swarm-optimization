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


data = preprocessData()

particle_n = 10
population = []

print("building..")
for p in range(particle_n):
    inputLayer = buildLayer(node=8)
    hiddenLayers = [buildLayer(node=np.random.randint(3, 6))
                    for i in range(np.random.randint(1, 3))]
    outputLayer = buildLayer(node=1)
    print(f'p : {p}', end=" ")
    particle = Model()
    particle.create(inputLayer, hiddenLayers, outputLayer)
    print(
        f'shape :{len(particle.inputLayer)} - {[len(l) for l in particle.hiddenLayers]} - {len(particle.outputLayer)}')
    population.append(particle)


time.sleep(3)


def velocity_vector(xpbest, xi, vi):
    return vi + (np.random.uniform(0, 1)*(xpbest - xi))


print('training..')
t_max = 25

cross_data = cross_validation_split(cross_validate_num=0.1, dataset=data)
block = cross_data["data_block"]
rand_set = cross_data["rand_set"]
reminder_set = cross_data["rem_set"]

cross_validation_plot = []
for c in range(10):
    res = select_validate(block, rand_set, c, reminder_set)
    train = res["train"]
    cross_valid = res["cross_valid"]
    wins = []
    printProgressBar(0, t_max, prefix='evolutioning',
                     suffix='', length=25)
    tic = time.perf_counter()
    for t in range(t_max):
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

        printProgressBar(
            t+1, t_max, prefix=f'evolutioning.. t={t+1} ,cross_validation : {c+1}', length=25)
    # cross validation
    best_validation = []
    for p in population:
        mae = p.evaluate(cross_valid)
        best_validation.append(mae)
    cross_validation_plot.append(np.min(best_validation))

    toc = time.perf_counter()
    print(f'pbest in all gen = {np.min(wins)}')
    if toc-tic >= 60:
        print(f"total execution time {((toc - tic)/60):0.2f} m\n")
    else:
        print(f"total execution time {toc - tic:0.4f} secs\n")


plt.plot(cross_validation_plot)
plt.title('cross validation')
plt.show()
