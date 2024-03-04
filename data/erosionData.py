import os
import os.path
import imageio
import matplotlib.pyplot as plt
import numpy as np


class ErosionData:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        if self.data_directory[-1] != '\\':
            self.data_directory += '\\'
        self.images = []
        for dirpath, dirnames, filenames in os.walk(data_directory):
            for filename in [f for f in filenames if f.startswith('Input')]:
                self.images.append(filename)
        self.currentImageIdx = 0
        self.inputImage = []
        self.outputImage = []

    def __extractChunk(self, image):
        return image[384:640, 384:640,:]

    def iterateImage(self):
        self.currentImageIdx += 1

    def loadImage(self, offset):
        if self.currentImageIdx >= len(self.images):
            return False
        filepath = self.data_directory + self.images[self.currentImageIdx+offset]
        inputImage = self.__extractChunk(imageio.imread(filepath))
        outputImage = self.__extractChunk(imageio.imread(filepath.replace('Input', 'Output')))
        self.inputImage = self.__imgNormalize(inputImage)
        self.outputImage = self.__imgNormalize(outputImage)
        return True
    
    def __imgNormalize(self, image):
        return image.astype(np.double) / 65535.0 * 2 - 1

    def __imgDenormalize(self, image):
        return np.multiply(np.divide(np.add(image, 1),2), 65535.0).astype(np.uint16)

    def saveImage(self, directory):
        inputImage = self.__imgDenormalize(self.inputImage)
        outputImage = self.__imgDenormalize(self.outputImage)
        inputImage = self.inputImage
        outputImage = self.outputImage
        imageio.imwrite(directory + self.images[self.currentImageIdx], inputImage)
        imageio.imwrite(directory + self.images[self.currentImageIdx].replace('Input', 'Output'), outputImage)

    def placeDEMInOriginalAndSaveAsDem(self, destinationDirectory):
        input_1024 = imageio.imread(self.data_directory+self.images[self.currentImageIdx])
        outputDenormalized = self.__imgDenormalize(self.outputImage)
        for i in range(384, 640):
            for j in range(384, 640):
                input_1024[i][j][0] = outputDenormalized[i-384][j-384][0]

        r_channel = input_1024[:, :, 0]
        imageio.imwrite(destinationDirectory+self.images[self.currentImageIdx], r_channel)
    
    def getImages(self):
        return self.inputImage, self.outputImage
    
    def size(self):
        return len(self.images)

    def setIndex(self, idx):
        self.currentImageIdx = idx
