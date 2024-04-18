import numpy as np
import argparse
import math

def random_sentence_from_m2(filename, percent):
  """
  This function randomly selects a sentence from an M2 file.

  Args:
      filename: The path to the M2 file.

  Returns:
      A tuple containing the original sentence and a list of edit annotations 
      (or None if the file is empty).
  """
  sentences = []
  current_sentence = []
  current_edits = []
  sentCnt = 0

  with open(filename, 'r') as f:
    for line in f:
      line = line.strip()  # Remove leading/trailing whitespace

      if not line:  # Empty line, signifies end of sentence
        if current_sentence:
          sentences.append((current_sentence, current_edits))
          current_sentence = []
          current_edits = []
      else:
        # Split line based on spaces (might need adjustment for different delimiters)
        # parts = line.split()
        if line.split()[0].upper() == 'S':
          current_sentence.append(line)  # Original sentence word
        else:
          # Extract edit information (assuming basic M2 format)
          current_edits.append(line)
        
        sentCnt += 1
        

  percent = 100.0 if (percent > 100.0 ) else percent
  percent = 1.0 if (percent<1.0) else percent
  res = math.floor(sentCnt * (percent/100.0))
  # print("taking",res,"from",sentCnt)
  
  sentSet = []

  # Randomly select a sentence (if any)

  if sentences:
    allIndices = range(len(sentences))
    selectedTestIndices = np.random.choice(allIndices, res)
    trainIndices = np.setdiff1d(allIndices, selectedTestIndices)
    testSet = [sentences[i] for i in selectedTestIndices]
    trainSet = [sentences[i] for i in trainIndices]
  #   for i in range(res):
  #     sentSet.append(random.choice(sentences))
      
    return trainSet, testSet #random.choice(sentences)
  else:
    return None
def main(args):

  filename = args.m2 #"m2/ABC.train.gold.bea19.m2"
  trainSet, testSet = random_sentence_from_m2(filename,args.percent)

  # if len(testSet):
  testSentences = ['\n'.join(sentence+edits) for sentence, edits in testSet]
  trainSentences = ['\n'.join(sentence+edits) for sentence, edits in trainSet]

  trainContent = '\n\n'.join(trainSentences)
  testContent = '\n\n'.join(testSentences)
  with open('./datasets_binary/train.m2', 'w') as f:
    f.write(trainContent)
  
  with open('./datasets_binary/test.m2', 'w') as f:
    f.write(testContent)

    # for i in range(len(testSet)):
    #   sentence, edits = testSet[i]
    #   print(sentence[0])
    #   for edit in edits:
    #     print(edit)
    #   print("\n")
  #   with open('')
  # else:
  #   print("M2 file is empty or corrupt.")
  
  
def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--m2', required=True)
  parser.add_argument('--percent', type=float, default=10.0)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = get_parser()
  main(args)
