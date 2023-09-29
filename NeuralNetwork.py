import numpy as np
import cv2
from emnist import extract_training_samples
from emnist import extract_test_samples
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from cv2.ximgproc import thinning



HIDDEN_LAYER_1 = 512
HIDDEN_LAYER_2 = 256
DROPOUT = 0.5
LEARN_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10

def get_mlp_model(hiddenLayerOne=HIDDEN_LAYER_1, hiddenLayerTwo=HIDDEN_LAYER_2, dropout=DROPOUT, learnRate=LEARN_RATE):
    # initialize a sequential model and add layer to flatten the input data
    model = Sequential()

    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dropout(dropout))
    # add two hidden layer with activation function relu and a dropout
    model.add(Dense(hiddenLayerOne, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(hiddenLayerTwo, activation="relu"))
    model.add(Dropout(0.2))
    # add a softmax layer on top
    model.add(Dense(num_classes, activation="softmax"))

    # compile the model
    model.compile(
        optimizer=Adam(learning_rate=learnRate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    # return compiled model
    return model



print("[INFO] downloading EMNIST...")
# the data, split between train and test sets
trainData_og, trainLabels_og = extract_training_samples('balanced')
testData_og, testLabels_og = extract_test_samples('balanced')

num_classes = 8
class_names = ['0', '1', '2', '3', '4', 'E', 'S/s', 'X/x']

# we keep only the values that we actually use in our problem
# values associated with each of the class in class_names
train_filter = np.where((trainLabels_og == 0 ) | (trainLabels_og == 1) | (trainLabels_og == 2) | (trainLabels_og == 3) |
                        (trainLabels_og == 4) | (trainLabels_og == 14) | (trainLabels_og == 28) | (trainLabels_og == 33))
test_filter = np.where((testLabels_og == 0) | (testLabels_og == 1) | (testLabels_og == 2) | (testLabels_og == 3) |
                       (testLabels_og == 4) | (testLabels_og == 14) | (testLabels_og == 28) | (testLabels_og == 33))

# filter out data and classes not used
trainData, trainLabels = trainData_og[train_filter], trainLabels_og[train_filter]
testData, testLabels = testData_og[test_filter], testLabels_og[test_filter]

#adjusting the values associated with the three remaining letters from the dataset (E, S, X)
# 14 -> E
train_modify = np.where((trainLabels == 14))
trainLabels[train_modify] = 5
test_modify = np.where((testLabels == 14))
testLabels[test_modify] = 5
# 28 -> S
train_modify = np.where((trainLabels == 28))
trainLabels[train_modify] = 6
test_modify = np.where((testLabels == 28))
testLabels[test_modify] = 6
# 33 -> X
train_modify = np.where((trainLabels == 33))
trainLabels[train_modify] = 7
test_modify = np.where((testLabels == 33))
testLabels[test_modify] = 7

print(f'We have {trainData.shape[0]} train samples')
print(f'We have {testData.shape[0]} test samples')

# operations to normalize the method later used to read new images
for i in range(len(trainData)):
    trainData[i] = thinning(trainData[i], cv2.ximgproc.THINNING_ZHANGSUEN)
    trainData[i] = cv2.dilate(trainData[i], cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)), iterations=1)
for i in range(len(testData)):
    testData[i] = thinning(testData[i], cv2.ximgproc.THINNING_ZHANGSUEN)
    testData[i] = cv2.dilate(testData[i], cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)), iterations=1)

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# stop earlier the training if it doesn't improve (the loss doesn't decrease)
keras_callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=0.001)]

# initialize model
print("[INFO] initializing model...")
model = get_mlp_model()

'''
# for debugging
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for i in range(30):
    plt.subplot(10, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainData[i], cmap=plt.cm.binary)
    plt.xlabel(trainLabels[i])
plt.show()
'''

# train the network
print("[INFO] training model...")
H = model.fit(x=trainData, y=trainLabels, validation_split=0.15, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=keras_callbacks)

# make predictions on the test set and evaluate it
print("[INFO] evaluating network...")
accuracy = model.evaluate(testData, testLabels)[1]
print("accuracy: {:.2f}%".format(accuracy * 100)) # accuracy percentage

model.save('handwritten_recognition.model') # save the model so that it can be load from other files

'''
import matplotlib.pyplot as plt

#plot loss curve
plt.figure(figsize=[6,4])
plt.plot(H.history['loss'], 'black', linewidth=2.0)
plt.plot(H.history['val_loss'], 'blue', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
plt.show()

#plot accuracy curve
plt.figure(figsize=[6,4])
plt.plot(H.history['accuracy'], 'black', linewidth=2.0)
plt.plot(H.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)
plt.show()
'''

'''
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier

# cross validation
cross_val_value = cross_val_score(KerasClassifier(model), testData, testLabels, cv=10, scoring='accuracy')
print("cross validation: ")
print(cross_val_value)
print("Cross-validation Mean: " + str(np.mean(cross_val_value)))
'''

'''
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# confusion matrix
predLabels = model.predict(testData)
predLabels = np.argmax(predLabels, axis=1)
result = confusion_matrix(testLabels, predLabels, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=result, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()
'''