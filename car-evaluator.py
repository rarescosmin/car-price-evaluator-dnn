import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from csv import DictReader
from csv import DictWriter
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class CustomDataSet(Dataset):
    def __init__(self, file_name):
        file_out = pd.read_csv(file_name)
        X = file_out.iloc[0:, 0:-1].values
        Y = file_out.iloc[0:, -1].values

        self.X_train = torch.tensor(X)
        self.Y_train = torch.tensor(Y)
    
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1*6, 1)
        self.fc2 = nn.Linear(1*6, 1)
        self.fc3 = nn.Linear(1*6, 1)
        self.fc4 = nn.Linear(1*6, 1)

#
#   gets SET with text as input and returns dictionary with {key: text, value: numerical value}
#
def parseTextData(input_set):
    computedDict = {}
    sorted_set = sorted(input_set)
    index = 0
    for input_set_value in sorted_set:
        computedDict[input_set_value] = index
        index = index + 1

    return computedDict

#
#   transforms carMake, carModel and carFuelType from word to number
#   each will be given a number from 0 to ...
#   this method will generate 2 csvs with the desired output
#


def generateMachineLearningData(input_file_name, output_file_name):

    car_makes_set = set({})
    car_models_set = set({})
    car_fuel_types_set = set({})

    car_makes_dict = {}
    car_models_dict = {}
    car_fuel_types_dict = {}

    with open(input_file_name, 'r') as read_obj:
        csv_reader = DictReader(read_obj)

        for row in csv_reader:
            car_makes_set.add(row['make'])
            car_models_set.add(row['model'])
            car_fuel_types_set.add(row['fuelType'])

        car_makes_dict = parseTextData(car_makes_set)
        car_models_dict = parseTextData(car_models_set)
        car_fuel_types_dict = parseTextData(car_fuel_types_set)

    

    with open(input_file_name, 'r') as read_obj:
        csv_reader = DictReader(read_obj)

        with open(output_file_name, 'a', newline='') as writer_obj:
            csv_writer = csv.DictWriter(writer_obj, fieldnames=['make', 'model', 'year', 'mileage', 'fuelType', 'engineCapacity', 'price'])
            csv_writer.writeheader()
            for row in csv_reader:
                row['make'] = car_makes_dict[row['make']]
                row['model'] = car_models_dict[row['model']]
                row['fuelType'] = car_fuel_types_dict[row['fuelType']]
                csv_writer.writerow(row)

#
#   plots car makes if more than treshhold number of cars are listed
#
def plotMostListedCars(treshhold):
    cars_data = pd.read_csv('./clean-csv-data/cars_train.csv', encoding='latin-1')
    car_makes = cars_data['make']

    car_makes_set = set({})
    for make in car_makes:
        car_makes_set.add(make)

    plot_data = {}
    for car_set_value in car_makes_set:
        if 'audi'.casefold() == car_set_value.casefold() or 'bmw'.casefold() == car_set_value.casefold():
            print('yes')
        count = 0
        for make in car_makes:
            if car_set_value == make:
                count = count + 1
        if count > treshhold:
            plot_data[car_set_value] = count

    plot_data_car_makes = list(plot_data.keys())
    plot_data_car_makes_count = list(plot_data.values())

    objects = plot_data_car_makes
    y_pos = np.arange(len(objects))
    performance = plot_data_car_makes_count

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Number of cars listed')
    plt.xlabel('Car makes')
    plt.title('Most listed cars on Autovit')
    plt.autoscale()
    plt.show()


def evaluateCarPriceWithNN():
    cars_data_frame = pd.read_csv('./train-test-data/cars_train.csv')
    print(cars_data_frame)
    

    
if __name__ == '__main__':
    #plotMostListedCars(500)
    #generateMachineLearningData('./clean-csv-data/cars_train.csv', './train-test-data/cars_train.csv')
    #generateMachineLearningData('./clean-csv-data/cars_test.csv', './train-test-data/cars_test.csv')
    evaluateCarPriceWithNN()
    print('SCRIPT FINISHED RUNNING!')
