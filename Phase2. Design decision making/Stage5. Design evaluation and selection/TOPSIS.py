import pandas as pd
import os
import numpy as np
import random

#TOPSIS selection process
def TOPSIS(design_id, support, time, stress, w_support, w_time, w_stress):

    # Step 1: Squared
    sqr_support = np.square(support)
    sqr_time = np.square(time)
    sqr_stress = np.square(stress)

    # Step 2: Normalization
    norm_support = (support / np.sqrt(np.sum(sqr_support)))
    norm_time = (time / np.sqrt(np.sum(sqr_time)))
    norm_stress = (stress / np.sqrt(np.sum(sqr_stress)))

    # Step 3: weight multiply
    weighted_support = w_support * norm_support
    weighted_time = w_time * norm_time
    weighted_stress = w_stress * norm_stress

    # Step 4: Ideal solution
    PIS_support = np.min(weighted_support)
    NIS_support = np.max(weighted_support)
    PIS_time = np.min(weighted_time)
    NIS_time = np.max(weighted_time)
    PIS_stress = np.min(weighted_stress)
    NIS_stress = np.max(weighted_stress)

    PIS_support_dist = np.square(PIS_support - weighted_support)
    NIS_support_dist = np.square(NIS_support - weighted_support)
    PIS_time_dist = np.square(PIS_time - weighted_time)
    NIS_time_dist = np.square(NIS_time - weighted_time)
    PIS_stress_dist = np.square(PIS_stress - weighted_stress)
    NIS_stress_dist = np.square(NIS_stress - weighted_stress)

    # Step 5: Closeness
    PIS_dist = np.sqrt(np.sum([PIS_support_dist, PIS_time_dist, PIS_stress_dist], axis=0))
    NIS_dist = np.sqrt(np.sum([NIS_support_dist, NIS_time_dist, NIS_stress_dist], axis=0))
    closeness = NIS_dist / (PIS_dist + NIS_dist)
    closeness_list = closeness.tolist()
    # print(PIS_dist)
    # print(NIS_dist)
    # print(closeness)
    # print(closeness_list)

    # Step 6: Rank
    rank = zip(design_id, closeness_list)
    rank1 = max(rank, key=lambda x: x[1])
    # print(rank)
    # print(rank1)
    rank1_list =  list(rank1)
    return rank1_list[0],rank1_list[1], rank1_list


TOPSIS_PATH = r'C:\Users\USER\PycharmProjects\Generative_design\TOPSIS'
PATH = os.path.join('./prediction results/prediction_results.xlsx')
# print(PATH)

df = pd.read_excel(PATH)

design_id = df['Design'].tolist()
support = df['Support'].to_numpy()
time = df['Time'].to_numpy()
stress = df['Stress'].to_numpy()

design_id_list = list()

w_support_list=list()
w_time_list=list()
w_stress_list=list()

closeness_list = list()
rank1_list = list()

# Construct the efficient frontier
for _ in range(100000):
    weights = np.random.dirichlet([1, 1, 1])
    w_support, w_time, w_stress = weights


    # w_support_list.append(w_support)
    # w_time_list.append(w_time)
    # w_stress_list.append(w_stress)

    print('w_support:',w_support)
    print('w_time:',w_time)
    print('w_stress:',w_stress)

    _, _, rank1 = TOPSIS(design_id, support, time, stress, w_support, w_time, w_stress)
    rank1.append(w_support)
    rank1.append(w_time)
    rank1.append(w_stress)
    rank1_list.append(rank1)
    # rank1_list.append(w_support)
    # rank1_list.append(w_time)
    # rank1_list.append(w_stress)
print(rank1_list)
rank1_df = pd.DataFrame(rank1_list, columns=['Design ID','Closeness','w_support','w_time','w_stress'])
rank1_df.to_excel(os.path.join(TOPSIS_PATH,'efficient_frontier.xlsx'))





