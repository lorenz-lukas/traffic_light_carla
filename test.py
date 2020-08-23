import os
import pandas


file_csv1 = pandas.read_csv("dataset/carla_all.csv")
# file_csv2 = pandas.read_csv("dataset/carla_all.csv")
deleted = []


BASE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(BASE_DIR, 'dataset', 'images')
files_name = [f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]

found = False

print(len(file_csv1))
print(len(files_name))

for i in range(0, len(file_csv1)):
    for j in range(0, len(files_name)): #len(files_name)
        if(file_csv1.iloc[i]['filename'] == files_name[j]):
            #print(files_name[j])
            found = True
            break

    if(not found):
        deleted.append(file_csv1.index[i])
        # print("Removed: {}".format(file_csv1.index[i]))  
    found = False

print(len(deleted))
# i = 0
# while(len(deleted)):
#     if(i == deleted[i]):
#         file_csv1 = file_csv1.drop(file_csv1.index[i])
#         print("Removed: {}".format(file_csv1.index[i]))
#     i += 1

print(len(file_csv1))    
file_csv1.to_csv("dataset/carla_all_2.csv", index=False, header=True)