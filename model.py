import numpy as np
import tensorflow as tf
import nibabel as nb
import keras.backend as K
import os

# no, am i not going to use regex.
img1 = nb.load("BETA_Subject001_Condition001_Source001.nii")
img2 = nb.load("BETA_Subject002_Condition001_Source001.nii")
img3 = nb.load("BETA_Subject003_Condition001_Source001.nii")
img4 = nb.load("BETA_Subject004_Condition001_Source001.nii")
img5 = nb.load("BETA_Subject005_Condition001_Source001.nii")
img6 = nb.load("BETA_Subject006_Condition001_Source001.nii")
img7 = nb.load("BETA_Subject007_Condition001_Source001.nii")
img8 = nb.load("BETA_Subject008_Condition001_Source001.nii")
img9 = nb.load("BETA_Subject009_Condition001_Source001.nii")
img10 = nb.load("BETA_Subject010_Condition001_Source001.nii")
img11 = nb.load("BETA_Subject011_Condition001_Source001.nii")
img12 = nb.load("BETA_Subject012_Condition001_Source001.nii")
img13 = nb.load("BETA_Subject013_Condition001_Source001.nii")
img14 = nb.load("BETA_Subject014_Condition001_Source001.nii")
img15 = nb.load("BETA_Subject015_Condition001_Source001.nii")
img16 = nb.load("BETA_Subject016_Condition001_Source001.nii")
img17 = nb.load("BETA_Subject017_Condition001_Source001.nii")
img18 = nb.load("BETA_Subject018_Condition001_Source001.nii")
img19 = nb.load("BETA_Subject019_Condition001_Source001.nii")
img20 = nb.load("BETA_Subject020_Condition001_Source001.nii")

# put all of the patients into an array so i can iterate over them later
img_list = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16, img17, img18, img19, img20]
brain_list = []
testing_list = []

def sig_voxels(img_array):
    # took the 5 most significant clusters and took the fc value from each patient's file
    # the parameter is an array because i translate the patient's fc data into an array later on
    # had to divide by two and subtract/add from (45, 54, 45) because of file compatibility between the fc analysis files and cluster analysis file
    return [img_array[45 + 15][54 - 21][45 + 25], img_array[45 + 24][54 - 6][45 + 8], img_array[45 - 32][54 - 6][45 + 3], img_array[45 - 13][54 - 23][45 + 27], img_array[45 - 28][54 - 17][45 + 28]]

# populates both arrays with the significant fc values from the clusters
def populate_brain_and_test_list():
    for i in range(0, 20):
        if(0 <= i < 5 or 10 <= i < 15):
            brain_list.append(sig_voxels(np.array(img_list[i].dataobj)))
    for i in range(0, 20):
        if(5 <= i < 10 or 15 <= i < 20):
            testing_list.append(sig_voxels(np.array(img_list[i].dataobj)))

populate_brain_and_test_list()

# boilerplate stuff to run the model
classes = ['non_adhd', 'adhd']
brain_array = np.array(brain_list)
labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
testing_array = np.array(testing_list)
testing_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# setting up the model architecture
seeds = tf.keras.layers.Input(shape=(5, ))
dense1 = tf.keras.layers.Dense(64, activation="relu")
dense2 = tf.keras.layers.Dense(64, activation="relu")
outputs = tf.keras.layers.Dense(2, activation='softmax')
adhd_model = tf.keras.Sequential()
adhd_model.add(seeds)
adhd_model.add(dense1)
adhd_model.add(dense2)
adhd_model.add(outputs)
adhd_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='SGD',
    metrics=["accuracy"],
)

# training
adhd_model.fit(brain_array, labels, batch_size=5, epochs=100, shuffle=True, verbose=2)

# testing
test_loss, test_acc = adhd_model.evaluate(testing_array,  testing_labels, verbose=2)

# so we can actually know the accuracy. without this line, the past 4 months of work would have been simply for display
# the average accuracy seems to be 70%, but it goes up and down slightly in each run
print('\nModel accuracy:', test_acc)
