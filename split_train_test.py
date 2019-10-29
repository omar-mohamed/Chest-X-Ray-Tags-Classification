import pandas as pd
import os
import numpy as np

dataset_df = pd.read_csv('./IU-XRay/all_data_tags.csv')

number_of_testing_cases=500
shuffle=True

if shuffle:
    dataset_df = dataset_df.sample(frac=1., random_state=np.random.randint(1,100))

training_df=dataset_df.head(-number_of_testing_cases)

print(training_df.head())

testing_df=dataset_df.tail(number_of_testing_cases)

print(testing_df.head())

training_df.to_csv(os.path.join("./IU-XRay","training_set.csv"), index=False)

testing_df.to_csv(os.path.join("./IU-XRay","testing_set.csv"), index=False)