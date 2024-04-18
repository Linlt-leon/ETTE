import json
import matplotlib.pyplot as plt

files = [
    './models/binary/log.json',
    './models/multi/log.json'
]

for file in files:
    with open(file, 'r') as fileReader:
        data = fileReader.read()
    
    modelType = file.split('/')[2]

    try:
        allEpochData = json.loads(data)
        upData = {}
        for epoch, log in allEpochData.items():
            eNum = int(epoch.split(' ')[1])
            upData[eNum] = log
        trainLosses = []
        validLosses = []
        f05Scores = []

        for i, (epoch, logs) in enumerate(sorted(upData.items())):
            trainLosses.append(logs['train_log']['loss'])
            validLosses.append(logs['valid_log']['loss'])
            f05Scores.append(logs['valid_log']['cls_report']['f05-score'])

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(trainLosses) + 1), trainLosses, label='Training Loss')
        plt.plot(range(1, len(validLosses) + 1), validLosses, label='Validation Loss')
        plt.plot(range(1, len(f05Scores) + 1), f05Scores, label='Macro-Average F0.5 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Loss/Score')
        plt.title('Training and Validation Loss with F0.5 score vs Epoch for {} Classfication'.format(modelType.title()))
        plt.legend()
        plt.savefig('{}.png'.format(modelType))
        plt.close()

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")