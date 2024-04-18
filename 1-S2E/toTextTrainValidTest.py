import os
import subprocess

dirPath = 'datasets_binary'
outputDirPaths = ['./datasets_binary/text', './datasets_multi/text']

preprocessScript = 'token_ged/preprocess.py'

def generateFilePath(dirPath, fileName):
    return "{}/{}".format(dirPath, fileName)

def generateSrcOption(inputFile):
    return "--m2 {}".format(inputFile)

def generateOutputFilePath(fileName, outputDirPath):
    return "{}/{}.txt".format(outputDirPath, fileName.split('.')[0])

for fileName in os.listdir(dirPath):
    filePath = generateFilePath(dirPath, fileName)
    if os.path.isfile(filePath):
        lines = open(filePath).read().rstrip().split('\n')
        testLines = '\n'.join(
            [' '.join(line.split(' ')[1:]) for line in lines if len(line.split(' ')) > 1 and line.split(' ')[0] == 'S']
        )

        for path in outputDirPaths:
            if not os.path.exists(path):
                os.makedirs(path)
            with open(generateOutputFilePath(fileName, path), 'w') as file:
                file.write(testLines)