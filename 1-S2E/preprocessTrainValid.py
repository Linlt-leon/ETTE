import os
import subprocess
import argparse

def main(dirPath, outputDirPath, mode):
    # dirPath = 'datasets'
    # outputDirPath = './datasets/preProcessed'

    preprocessScript = 'token_ged/preprocess.py'

    def generateFilePath(dirPath, fileName):
        return "{}/{}".format(dirPath, fileName)

    def generateSrcOption(inputFile):
        return "--m2 {}".format(inputFile)

    def generateOutputFilePath(fileName):
        return "{}/{}json".format(outputDirPath, fileName[:-2])

    for fileName in os.listdir(dirPath):
        filePath = generateFilePath(dirPath, fileName)
        if os.path.isfile(filePath):
            command = ['python', preprocessScript]
            command.extend(['--m2', filePath, '--mode', mode, '--out', generateOutputFilePath(fileName)])
            print(command)
            if not os.path.exists(outputDirPath):
                os.makedirs(outputDirPath)
            subprocess.run(command, check=True)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--outDir', required=True)
    parser.add_argument('--mode', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args.dir, args.outDir, args.mode)