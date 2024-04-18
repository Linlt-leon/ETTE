import os

dirPath = './m2'

def generateFilePath(dirPath, fileName):
    return "{}/{}".format(dirPath, fileName)

def removeNonUtfCharacters(inputFile):
    fileContent = None
    with open(inputFile, 'r', encoding='utf-8') as file:
        fileContent = file.read()
        
    cleaned_content = ''.join(c if ord(c) < 128 else '?' for c in fileContent)

    with open(inputFile, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)

for fileName in os.listdir(dirPath):
    removeNonUtfCharacters(generateFilePath(dirPath, fileName))
    
