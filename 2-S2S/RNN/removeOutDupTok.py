from tqdm import tqdm

def eliminate_repeated_tokens_list(fname):
  """
  Eliminates repeated tokens in a txt list chronologically using a list.

  Args:
      filename: The path to the txt file.

  Returns:
      A list of unique tokens in the order they appeared in the file.
  """
  
  out_lines = []
  with open(fname, 'r') as f:
  # with tqdm(total=4382) as progress_bar:
    for line in f:
      unique_tokens = []
      tokens = line.strip().split()
      # print('>>>', len(out_lines)+1)
      # print(tokens)
      if (len(tokens)!=0):
        unique_tokens.append(tokens[0])
        for tok in tokens:
          if (tok and tok != unique_tokens[-1]) and \
            (len(unique_tokens)<2) :
            unique_tokens.append(tok)
          else:
            if len(unique_tokens)>=2 and (tok != unique_tokens[-1] and tok != unique_tokens[-2]) :  
              unique_tokens.append(tok)  
              
      out_lines.append(unique_tokens)
      
    # progress_bar.update(1)

  return out_lines

# Example usage 
fname = 'train.105-rnn.out2' 
out_lines = eliminate_repeated_tokens_list(fname)

for line in out_lines:
  if len(line)> 0:
    print(' '.join(line))
  else:
    print(' ')