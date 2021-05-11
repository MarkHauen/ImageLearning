from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import ImageProcessor as ip

def convertSciToFloat(numAsStr):
    try:
        if ('e' in numAsStr) & (len(numAsStr) > 0):
            x = numAsStr.split('e')
            return float(x[0]) * (10 ** float(x[1]))
        else:
            return float(numAsStr)
    except ValueError as e:
        return 0

def formatPredictions(predictions):
    predictionsAsString = ""
    for x in predictions:
        for y in x:
            predictionsAsString = predictionsAsString + str(y) + ","
        predictionsAsString = predictionsAsString + "|"
    return predictionsAsString

train_images = ip.train_nparray
train_labels = ip.train_labels
test_images = ip.validation_nparray
test_labels = ip.validation_labels

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(168, 300, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n\nTest accuracy: %{0}\n\n'.format(test_acc * 100))


def runTest():
    print("Test should find two Closed and one Open")
    predictions = model.predict(test_images)
    predictionsAsString = formatPredictions(predictions)
    predictionsStringList = predictionsAsString.split("|")
    predictions = []
    testPredictionsIndex = []
    for x in predictionsStringList:
        predictions.append([convertSciToFloat(y) for y in x.split(",")])
    predictions = [x for x in predictions if len(x) > 1]
    for x in predictions:
        testPredictionsIndex.append(x.index(max(x)))
    for x in testPredictionsIndex:
        print(["Closed", "Open"][x])
