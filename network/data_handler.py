from numpy.random import permutation
from numpy import array

def split_data(data, r1=0.8, r2=0.2):
    length = len(data)
    # shuffle data
    idx = permutation(len(data))
    data = [data[i] for i in idx]

    pin1 = int(length * r1)
    pin2 = pin1 + int(length * r2)
    pin3 = pin2 + int(length * (1 - r2 - r1))

    training_set = data[:pin1]
    testing_set = data[pin1:pin2]
    validation_set = data[pin2:pin3]

    return training_set, testing_set, validation_set

def read_csv(file:str, dtype=int):
    f = file.read().splitlines()
    categ = f[0].split(',')
    rows = [[dtype(point) for point in row.split(',')] for row in f[1:]]
    return array(categ, dtype=str), array(rows, dtype=dtype)
