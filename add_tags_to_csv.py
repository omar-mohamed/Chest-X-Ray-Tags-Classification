from configs import argHandler  # Import the default arguments
import pandas as pd
import os

FLAGS = argHandler()
FLAGS.setDefaults()

classes = FLAGS.classes

tags = []
labeler_csv=pd.read_csv("IU-XRay/all_data_labeler_output.csv")
all_data_csv=pd.read_csv("IU-XRay/all_data.csv")

for i, row in labeler_csv.iterrows():
    tag=''
    index=0
    for elem in row[1:]:
        if elem == 1:
            if tag != '':
                tag += ','
            tag += classes[index]
        index +=1
    if tag == '':
        tags.append('No Finding')
    else:
        tags.append(tag)

all_data_csv['Tags']=tags

all_data_csv.to_csv(os.path.join("IU-XRay","all_data_tags.csv"), index=False)



