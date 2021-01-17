import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from csv import DictReader
from csv import DictWriter

#
#   gets SET with text as input and returns dictionary with {key: text, value: numerical value}
#
def parseTextData(input_set):
    computedDict = {}
    sorted_set = input_set.sort()
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

    with open(input_file_name, 'r') as read_obj:
        csv_reader = DictReader(read_obj)

        for row in csv_reader:
            car_makes_set.add(row['make'])
            car_models_set.add(row['model'])
            car_fuel_types_set.add(row['fuelType'])
        
        car_makes_dict = parseTextData(car_makes_set)
        car_models_set = parseTextData(car_models_set)
        car_fuel_types_set = parseTextData(car_fuel_types_set)

        

        with open(output_file_name, 'a', newline='') as writer_obj:
            csv_writer = csv.DictWriter(writer_obj, fieldnames=['make', 'model', 'year', 'mileage', 'fuelType', 'engineCapacity', 'price'])
            csv_writer.writeheader()
            for row in csv_reader:
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

    
if __name__ == '__main__':
    #plotMostListedCars(500)
    generateMachineLearningData('./clean-csv-data/cars_train.csv', './train-test-data/cars_train.csv')
    print('SCRIPT FINISHED RUNNING!')