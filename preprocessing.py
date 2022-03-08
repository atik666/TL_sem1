""" Code for preprocessing the imagenet data """
# Importing libraries
import pandas as pd
import os
import shutil

# Main directory
dir = "/home/huaxia/Documents/Atik/ImageNet-ILSVRC2012/"

# Opening the train image labels
train_file = open(dir + "map_clsloc.txt").read().split()
#Separating files based on class number and label
data_train = pd.DataFrame(columns = ["Folder_num", "Class_num", "Class_label"])
# Putting the files in the correct columns
for i in range(0,len(train_file),3):
    data_train = data_train.append({"Folder_num": train_file[i], "Class_num": train_file[i+1],"Class_label": train_file[i+2]}, ignore_index=True)

# Finding out the repeated classes
data_train.query("Class_label == 'crane'")
data_train["Class_label"][544] = 'crane_manmade'

data_train.query("Folder_num == 'n03710721'")
data_train.query("Class_label == 'maillot'")
data_train["Class_label"][976] = 'swimsuit'

# Rename folder based on the class labels
src = dir + "train/"

for i in range(len(data_train)):
    try:
        os.rename(src+data_train["Folder_num"][i], src+data_train["Class_label"][i])
    except:
        pass
    
# Put class label for the validation dataset
path = dir + "ILSVRC2016_devkit/ILSVRC/devkit/data"
    
val_file = open(dir + "ILSVRC2015_clsloc_validation_ground_truth.txt").read().split()

val_file = pd.DataFrame(val_file)

val_file["Class_label"] = "" # Creating empty labels for all validation data

# Put class label based on class number
for n in range(len(data_train)):
    for i in range(len(val_file)):
        if val_file["Class_label"][i] == "" and val_file[0][i] == data_train["Class_num"][n]:
            val_file["Class_label"][i] = data_train["Class_label"][n]

# Check if any labelling is missing         
val_file.query("Class_label == ''")

# Make validation folders for different labels
for n in range(len(data_train)):
    os.mkdir(dir+"val/"+data_train["Class_label"][n])

# Rename all validation files based on their class number
folder = "/home/huaxia/Documents/Atik/ImageNet-ILSVRC2012/ILSVRC2012_img_val"
for count, filename in enumerate(sorted(os.listdir(folder))):
    dst = f"{str(count)}.JPEG"
    src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{folder}/{dst}"
    os.rename(src, dst)

# Move validation files based on their class labels
for filename in (os.listdir(folder)):
    print(filename)
    name = os.path.splitext(filename)[0]           
    if str(val_file.index[int(name)]) == str(name):
        dst = f"{dir}val/{val_file['Class_label'][int(name)]}/{str(name)}.JPEG"
        shutil.move(folder+"/"+filename, dst)

# Partitioning A & B randomly
location = dir + 'ImageNet_Processed/'
files = os.listdir(location+"train/")

for count, filename in enumerate(files):
    try:
        while count < 500:
            loc_src = location+"train/"+filename+"/"
            loc_des = location+"A/train/"+filename+"/"
            loc_src1 = location+"val/"+filename+"/"
            loc_des1 = location+"A/val/"+filename+"/"
            shutil.copytree(loc_src, loc_des)
            shutil.copytree(loc_src1, loc_des1)
            count += 1
        else:
            loc_src = location+"train/"+filename+"/"
            loc_des = location+"B/train/"+filename+"/"
            loc_src1 = location+"val/"+filename+"/"
            loc_des1 = location+"B/val/"+filename+"/"
            shutil.copytree(loc_src, loc_des)
            shutil.copytree(loc_src1, loc_des1)
            count += 1
    except:
        print("exception occured")
        pass