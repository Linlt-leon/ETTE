import os
import subprocess

binaryDirPath = 'datasets_binary/text'
multiDirPath = 'datasets_multi/text'
binaryModelPath = 'models/binary/best/'
multiModelPath = 'models/multi/best/'
modelPaths = [binaryModelPath, multiModelPath]
outputDirPath = './predicted'
binaryOutputDir = '{}/{}'.format(outputDirPath, 'binary')
multiOutputDir = '{}/{}'.format(outputDirPath, 'multi')

if not os.path.exists(binaryOutputDir) and not os.path.exists(multiOutputDir) :
    os.makedirs(binaryOutputDir)
    os.makedirs(multiOutputDir)

predictScript = 'token_ged/predict.py'

def generateFilePath(dirPath, fileName):
    return "{}/{}".format(dirPath, fileName)

def generateOutputFilePath(fileName, cType = 'binary'):
    return "{}/{}/{}.txt".format(outputDirPath, cType, fileName.split('.')[0])

for fileName in os.listdir(binaryDirPath):
    filePath = generateFilePath(binaryDirPath, fileName)
    if os.path.isfile(filePath):
        command = [
            'accelerate', 
            'launch', 
            '--num_processes', 
            '8', 
            predictScript]
        command.extend(['--input', filePath, '--restore_dir', binaryModelPath, '--output', generateOutputFilePath(fileName, 'binary')])
        print(command)
        if not os.path.exists(outputDirPath):
            os.makedirs(outputDirPath)
        subprocess.run(command, check=True)

for fileName in os.listdir(multiDirPath):
    filePath = generateFilePath(multiDirPath, fileName)
    if os.path.isfile(filePath):
        command = ['accelerate', 'launch', '--num_processes', '4', predictScript]
        command.extend(['--input', filePath, '--restore_dir', multiModelPath, '--output', generateOutputFilePath(fileName, 'multi')])
        print(command)
        if not os.path.exists(outputDirPath):
            os.makedirs(outputDirPath)
        subprocess.run(command, check=True)