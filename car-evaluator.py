import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    plotMostListedCars(500)
    print('SCRIPT FINISHED RUNNING!')