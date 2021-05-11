from ImageProcessor import loadImage, np
from Predictor import model, runTest

def main():
    runTest()
    while True:
        path = input("Enter path of image to test")
        if path == "x":
            break
        testImage = np.array([loadImage(path)])
        prediction = model.predict(testImage)[0]
        prediction = [float(x) for x in prediction]
        prediction = prediction.index(max(prediction))
        print(["Closed", "Open"][prediction])

if __name__ == "__main__":
   main()