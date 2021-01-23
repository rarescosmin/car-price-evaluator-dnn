import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_csv_data(path, fieldnames):
    raw_dataset = pd.read_csv(path, names=fieldnames, encoding='latin-1')
    return raw_dataset



def create_soup(x):
    return x['make'] + ' ' + x['model'] + ' ' + str(x['year']) + ' ' + str(x['mileage']) + ' ' + str(x['fuelType']) + ' ' + str(x['price'])


def get_recommendations(metadata, indices, title, cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata.iloc[movie_indices]

def create_soup_2(x):
    return x['make']+ ' ' + x['model'] + ' ' + str(x['year']) + ' ' + str(x['price'])

if __name__ == '__main__':
    logging.basicConfig(filename='car-reco.log',
                        level=logging.DEBUG, filemode='w')
    
    raw_dataset = load_csv_data(
        path='./clean-csv-data/cars_train.csv',
        fieldnames=['make', 'model', 'year', 'mileage',
            'fuelType', 'engineCapacity', 'cylinders', 'price']
    )

    raw_dataset['soup'] = raw_dataset.apply(create_soup, axis=1)

    raw_dataset['soup_2'] = raw_dataset.apply(create_soup_2, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(raw_dataset['soup'])

    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    raw_dataset = raw_dataset.reset_index()
    indices = pd.Series(raw_dataset.index, index=raw_dataset['soup_2'])
    

    reccs = get_recommendations(raw_dataset, indices, 'audi a4 2006 4199', cosine_sim)

    print(reccs)