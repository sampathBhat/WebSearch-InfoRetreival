from pandas import Series
import math
import numpy as np
import pandas as pd

class TestUser:

    def __init__(self, user_id):
        self.user_id = user_id
        self.list_of_rated_movie = []
        self.list_of_rate_of_rated_movie = []
        self.list_of_unrated_movie = []

    def get_user_id(self):
        return self.user_id

    def get_list_of_rated_movie(self):
        return self.list_of_rated_movie

    def get_list_of_rate_of_rated_movie(self):
        return self.list_of_rate_of_rated_movie

    def get_list_of_unrated_movie(self):
        return self.list_of_unrated_movie

class UserTestTable:

    def __init__(self):
        self.map = {}

    def get_map(self):
        return self.map

    def put_user(self, user_id, test_user):
        self.map[user_id] = test_user

    def get_user(self, user_id):
        return self.map.get(user_id)


def compute_train_matrix(train_file, train_matrix):

    file = open(train_file, "r")
    lines_of_file = file.read().strip().split("\n")

    for i in range(len(lines_of_file)):
        line = lines_of_file[i]
        train_matrix[i] = [int(val) for val in line.split()]

def compute_iuf_data(train_matrix):
    iuf_map = {}

    num_row = len(train_matrix)
    num_col = len(train_matrix[0])

    for c_idx in range(num_col):
        iuf = 0.0
        movie_id = c_idx + 1
        total_num_of_users = num_row

        lst_rate = []

        for r_idx in range(num_row):
            user_id = r_idx + 1

            rate = train_matrix[user_id - 1][movie_id - 1]
            if rate > 0: lst_rate.append(rate)

        if len(lst_rate) != 0:
            iuf = math.log10(total_num_of_users / len(lst_rate))
        else:
            iuf = 1.0

        for r_idx in range(num_row):
            train_matrix[r_idx][c_idx] = iuf * train_matrix[r_idx][c_idx]

        iuf_map[movie_id] = iuf


def build_user_test_table(test_file):
    df = pd.read_table(test_file, sep="\s+", header=None)

    rows = df.shape[0]

    user_map = UserTestTable().get_map()

    for i in range(rows):
        user_id = df[0][i]

        if user_id not in user_map:
            user = TestUser(user_id)
            user_map[user_id] = user

        user = user_map[user_id]
        rate = df[2][i]
        if rate > 0:
            user.get_list_of_rated_movie().append(df[1][i])
            user.get_list_of_rate_of_rated_movie().append(df[2][i])
        else:
            user.get_list_of_unrated_movie().append(df[1][i])

    return user_map



def average_rating_trainMovies(train_matrix):
   

    map_mean_rate = {}

    t_train_matrix = np.array(train_matrix).T

    for index, row in enumerate(t_train_matrix):
        movie_id = index + 1
        mean_rate = 0.0

        non_zero = [rate for rate in row if rate > 0]
        if len(non_zero) > 0:
            mean_rate = sum(non_zero) / len(non_zero)

        map_mean_rate[movie_id] = mean_rate

    return map_mean_rate

def average_movieRate_testUser(user_id, test_map):
    user = test_map[user_id]
    list_of_rate_of_rated_movie = user.get_list_of_rate_of_rated_movie()

    avg_rate = 0.0
    if len(list_of_rate_of_rated_movie) != 0:
        avg_rate = sum(list_of_rate_of_rated_movie) / len(list_of_rate_of_rated_movie)

    return avg_rate

def average_rating_trainUsers(train_matrix):

    map_mean_train_users = {}
    for index, row in enumerate(train_matrix):
        mean_rate = 0.0

        user_id = index + 1
        non_zero_list = [rate for rate in row if rate > 0]

        if len(non_zero_list) > 0:
            mean_rate = sum(non_zero_list) / len(non_zero_list)

        map_mean_train_users[user_id] = mean_rate

    return map_mean_train_users

def compute_cosine_similarity(v1, v2):

    if len(v1) != len(v2) or len(v1) == 0 or len(v2) == 0:
        print("Error: invalid input, the length of vectors should be equal and larger than 0")
        return

    result = 0.0
    numerator = 0.0
    denominator = 0.0
    numerator = np.inner(v1, v2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denominator != 0.0:
        result = numerator / denominator

    return result

# Cosine Similarity
def top_neighbour_cosine(user_id, train_matrix, test_map):

    test_user = test_map[user_id]
    list_of_unrated_movie = test_user.get_list_of_unrated_movie()
    list_of_rated_movie = test_user.get_list_of_rated_movie()
    list_of_rate_of_rated_movie = test_user.get_list_of_rate_of_rated_movie()

    list_of_neighbor = []
    for row in range(len(train_matrix)):
        train_user_id = row + 1

        if train_user_id == user_id: continue

        common_movie = 0

        numerator = 0.0
        denominator = 0.0
        cosine_similarity = 0.0

        test_vector = []
        train_vector = []

        for i in range(len(list_of_rated_movie)):
            movie_id = list_of_rated_movie[i]
            test_movie_rate = list_of_rate_of_rated_movie[i]
            train_movie_rate = train_matrix[train_user_id - 1][movie_id - 1]

            if train_movie_rate != 0:
                common_movie += 1

                test_vector.append(test_movie_rate)
                train_vector.append(train_movie_rate)

        if common_movie > 1:
            cosine_similarity = compute_cosine_similarity(test_vector, train_vector)

            list_of_neighbor.append((train_user_id, cosine_similarity))

    list_of_neighbor.sort(key=lambda tup : tup[1], reverse=True)

    return list_of_neighbor


def cosine_similarity_prediction(user_id, movie_id, num_of_neighbor, train_matrix, test_map, list_of_neighbors, avg_train_movie):

    # average rate of user in the test data
    avg_movie_rate_in_test = average_movieRate_testUser(user_id, test_map)

    numerator = 0.0
    denominator = 0.0

    counter = 0
    for i in range(len(list_of_neighbors)):
        if counter > num_of_neighbor: break

        neighbor_id = list_of_neighbors[i][0]
        neighbor_similarity = list_of_neighbors[i][1]
        neighbor_movie_rate = train_matrix[neighbor_id - 1][movie_id - 1]

        if neighbor_movie_rate > 0:
            counter += 1

            # case amplification
            p = 2.5
            neighbor_similarity *= math.pow(math.fabs(neighbor_similarity), (p - 1))

            numerator += neighbor_similarity * neighbor_movie_rate
            denominator += neighbor_similarity

    if denominator != 0.0:
        result = numerator / denominator
    else:
        result = avg_movie_rate_in_test

    result = int(round(result))

    return result


# Pearson Correlation
def compute_pearson_correlation(test_users, test_mean_rate, train_users, train_mean_rate):
    numerator = 0.0
    denominator = 0.0

    # filter the common components as vector
    vector_test_rates = []
    vector_train_rates = []

    for test_movie_id, test_movie_rate in test_users:
        train_movie_rate = train_users[test_movie_id - 1]
        if train_movie_rate > 0 and test_movie_rate > 0:
            vector_test_rates.append(test_movie_rate)
            vector_train_rates.append(train_movie_rate)

    if len(vector_train_rates) == 0 or len(vector_test_rates) == 0:
        return 0.0

    adj_vector_test_users = [movie_rate - test_mean_rate for movie_rate in vector_test_rates]
    adj_vector_train_users = [movie_rate - train_mean_rate for movie_rate in vector_train_rates]

    numerator = np.inner(adj_vector_train_users, adj_vector_train_users)
    denominator = np.linalg.norm(adj_vector_test_users) * np.linalg.norm(adj_vector_train_users)

    if denominator == 0.0:
        return 0.0

    return numerator / denominator

def top_neighbour_pearson(user_id, train_matrix, test_map, avg_train_rating):
    list_of_neighbors = []

    avg_movie_rate_in_test = average_movieRate_testUser(user_id, test_map)

    user = test_map[user_id]
    list_of_rated_movie = user.get_list_of_rated_movie()
    list_of_rate_of_rated_movie = user.get_list_of_rate_of_rated_movie()
    list_of_unrated_movie = user.get_list_of_unrated_movie()

    zipped_list_of_rated_movie_with_rate = []
    for i in range(len(list_of_rated_movie)):
        zipped_list_of_rated_movie_with_rate.append((list_of_rated_movie[i], list_of_rate_of_rated_movie[i]))

    for index, row in enumerate(train_matrix):
        train_user_id = index + 1

        if train_user_id == user_id: continue
        avg_movie_rate_in_train = avg_train_rating[train_user_id]

        pearson_correlation = compute_pearson_correlation(zipped_list_of_rated_movie_with_rate, avg_movie_rate_in_test, row, avg_movie_rate_in_train)

        if pearson_correlation > 1.0:
            pearson_correlation = 1.0
        if pearson_correlation < -1.0:
            pearson_correlation = -1.0

        if pearson_correlation != 0.0:
            list_of_neighbors.append((train_user_id, pearson_correlation))

    return list_of_neighbors

def pearson_correlation_prediction(user_id, movie_id, train_matrix, test_map, avg_train_rating, list_of_neighbors):

    result = 0.0
    numerator = 0.0
    denominator = 0.0

    test_mean_rate = average_movieRate_testUser(user_id, test_map)

    for neighbor in list_of_neighbors:
        train_user_id = neighbor[0]
        pearson_correlation = neighbor[1]

        train_mean_rate = avg_train_rating[train_user_id]

        train_user_rate = train_matrix[train_user_id - 1][movie_id - 1]
        if train_user_rate > 0:

            # case amplification
            p = 2.5
            pearson_correlation *= math.pow(math.fabs(pearson_correlation), (p - 1))

            numerator += pearson_correlation * (train_user_rate - train_mean_rate)
            denominator += math.fabs(pearson_correlation)

    if denominator != 0.0:
        result = test_mean_rate + numerator / denominator
    else:
        result = test_mean_rate

    result = int(round(result))

    if result > 5:
        result = 5
    if result < 1:
        result = 1

    return result

def compute_adjusted_cosine_similarity(target_id, neighbor_id, t_train_matrix, avg_train_rating):
    
    adj_cosine_sim = 0.0

    target_row = t_train_matrix[target_id - 1]
    neighbor_row = t_train_matrix[neighbor_id - 1]

    target_vector = []
    neighbor_vector = []
    for i in range(len(t_train_matrix[0])):
        if target_row[i] > 0 and neighbor_row[i] > 0:
            target_vector.append(target_row[i] - avg_train_rating[i + 1])
            neighbor_vector.append((neighbor_row[i]) - avg_train_rating[i + 1])

    if len(target_vector) == len(neighbor_vector) and len(target_vector) > 1:
        adj_cosine_sim = compute_cosine_similarity(target_vector, neighbor_vector)

    return adj_cosine_sim

def top_neighbor_adj_cosine(train_matrix, avg_train_rating):
    
    neighbor_map = {}

    t_train_matrix = np.array(train_matrix).T

    for i in range(len(t_train_matrix)):
        target_id = i + 1

        if target_id not in neighbor_map:
            neighbor_map[target_id] = {}

        for j in range(i + 1, len(t_train_matrix)):
            neighbor_id = j + 1

            adj_cosine_sim = compute_adjusted_cosine_similarity(target_id, neighbor_id, t_train_matrix, avg_train_rating)

            neighbor_map[target_id][neighbor_id] = adj_cosine_sim

            if neighbor_id not in neighbor_map:
                neighbor_map[neighbor_id] = {}

            neighbor_map[neighbor_id][target_id] = adj_cosine_sim

    return neighbor_map

def item_based_adjusted_cosine_prediction(user_id, movie_id, test_map, neighbor_map, train_mean_map):

    result = 0.0
    numerator = 0.0
    denominator = 0.0

    user = test_map[user_id]
    list_of_rated_movie = user.get_list_of_rated_movie()
    list_of_rate_of_rated_movie = user.get_list_of_rate_of_rated_movie()

    test_user_mean_rate = average_movieRate_testUser(user_id, test_map)

    map_of_neighbors = neighbor_map[movie_id]

    for i in range(len(list_of_rated_movie)):
        rated_movie_id = list_of_rated_movie[i]
        rated_movie_rate = list_of_rate_of_rated_movie[i]

        if rated_movie_id in map_of_neighbors:
            adj_cosine_sim = map_of_neighbors[rated_movie_id]
        else:
            adj_cosine_sim = 0.0

        numerator += adj_cosine_sim * (rated_movie_rate - test_user_mean_rate)
        denominator += abs(adj_cosine_sim)

    if denominator != 0.0:
        result = test_user_mean_rate + numerator / denominator
    else:
        result = test_user_mean_rate

    result = int(round(result))

    #Just to be double sure!!
    if result > 5:
        result = 5
    if result < 1:
        result = 1

    return result

def compute(i_file,train_matrix,avg_train_rating,avg_train_movie,iuf_train_matrix,adj_cosine_map_of_neighbors):
    out_file = open(i_file[0], "w")

    test_map = build_user_test_table(i_file[1])
    num_of_neighbor = 100

    # sort the test users
    list_of_test_user_id = sorted(test_map.keys())


    for user_id in list_of_test_user_id:
        user = test_map[user_id]
        list_of_unrated_movie = user.get_list_of_unrated_movie()

        # neighbor searching based on cosine similarity
        cosine_list_of_neighbors = top_neighbour_cosine(user_id, train_matrix, test_map)

        # neighbor searching based on pearson correlation
        pearson_list_of_neighbors = top_neighbour_pearson(user_id, train_matrix, test_map, avg_train_rating)

        for movie_id in list_of_unrated_movie:
            # the predicted rating based on cosine similarity
            #cosine_rating = cosine_similarity_prediction(user_id, movie_id, num_of_neighbor, train_matrix, test_map, cosine_list_of_neighbors, avg_train_movie)

            # the predicted rating based on pearson correlation
            #pearson_rating = pearson_correlation_prediction(user_id, movie_id, train_matrix, test_map, avg_train_rating, pearson_list_of_neighbors)

            # the predicted rating based on item based adjusted cosine similarity
            #item_based_adj_cosine_rating = item_based_adjusted_cosine_prediction(user_id, movie_id, test_map, adj_cosine_map_of_neighbors, avg_train_rating)

            # Customized Algorithm
            customized_rating = int(round(0.35 * cosine_rating + 0.35 * pearson_rating + 0.3 * item_based_adj_cosine_rating))

            out_line = str(user_id) + " " + str(movie_id) + " " + str(customized_rating) + "\n"
            out_file.write(out_line)

    out_file.close()

def main():
#
#Uncomment lines 454 to 464 to execute different Algorithm.
#Ensure to change line 466 accordingly.
# 
    file_list = [("./result20.txt", "test20.txt"), ("./result10.txt", "test10.txt"), ("./result5.txt", "test5.txt")]
    train_file = "train.txt"

    user_no = 200
    movies_no = 1000
    train_matrix = [[0] * movies_no] * user_no
    compute_train_matrix(train_file, train_matrix)

    # build the iuf train matrix from train.txt
    iuf_train_matrix = [[0] * movies_no] * user_no
    compute_train_matrix(train_file, iuf_train_matrix)
    compute_iuf_data(iuf_train_matrix)

    avg_train_rating = average_rating_trainUsers(train_matrix)

    avg_train_movie = average_rating_trainMovies(train_matrix)

    # build neighbor map based on adjusted cosine similarity
    adj_cosine_map_of_neighbors = top_neighbor_adj_cosine(train_matrix, avg_train_rating)

    for i_file in file_list:
        compute(i_file,train_matrix,avg_train_rating,avg_train_movie,iuf_train_matrix,adj_cosine_map_of_neighbors)

main()