# Example M2 data (assuming space-separated elements)
m2_data = """S Pasteries are very common and most of them offer the special dessert from the city .
A 0 1|||UNK|||Pasteries|||REQUIRED|||-NONE-|||0
A 12 13|||R:PREP|||of|||REQUIRED|||-NONE-|||0

S There are a comercial zone along the widest street of the city where you can find all kind of establishments : banks , bars , chemists , cinemas , pet shops , restaurants , fast food restaurants , groceries , travel agencies , supermarkets and others .
A 1 2|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||0
A 3 4|||R:SPELL|||commercial|||REQUIRED|||-NONE-|||0
A 17 18|||R:NOUN:NUM|||kinds|||REQUIRED|||-NONE-|||0
A 19 20|||R:NOUN|||businesses|||REQUIRED|||-NONE-|||0
A 38 39|||R:NOUN|||grocers|||REQUIRED|||-NONE-|||0

S Most of the shops have sales and offers at least three months of the year : January , June and August .
A 8 8|||M:PREP|||in|||REQUIRED|||-NONE-|||0"""

def edit_sentence(current_sentence, current_edits):
    """
    This function generates target sentences (corrected versions) from sentence and edits.

    Args:
        current_sentence: source sentence.
        current_edits: edit list

    Returns:
        A target sentence (corrected version).
    """
  
    target_sentence = []
    src = current_sentence.split()[1:]
    s_idx = 0
    if current_edits[0].split()[1] == '-1':
      return ' '.join(current_sentence.split()[1:])
    
    if current_edits[0].split()[1] != '0':
      target_sentence.extend(src[ :int(current_edits[0].split()[1]) ])
      s_idx = int(current_edits[0].split()[1])

    for edit in current_edits:
      e = edit.split('|||')
      e_b, e_e = int(e[0].split()[1]), int(e[0].split()[2])
      e_tok = e[2]
      # print('Editing tok:', e_tok if len(e_tok)>=1 else 'Null')
 
      if s_idx == e_b:
        if len(e_tok)>=1:
          target_sentence.extend([e_tok])

      else:
        target_sentence.extend(src[s_idx:e_b])
        if len(e_tok)>=1:
          target_sentence.extend([e_tok])
      s_idx = e_e
    
    if s_idx != len(current_sentence):
      target_sentence.extend(src[s_idx:])
    
    
    return ' '.join(target_sentence)

def read_sentences(fname):
  """
  This function read sentences from M2 data.

  Args:
      fname: A string containing M2 formatted data file.

  Returns:
      A list of sentences .
  """
  sentences = []
  current_sentence = []
  current_edits = []

  with open(fname, 'r') as f:
    for line in f:
      l = line.strip()  # Remove leading/trailing whitespace

      if len(l) > 0:  # Empty line, signifies end of sentence
          # Split line based on spaces (might need adjustment for different delimiters)

        if l.split()[0].upper() == 'S':
          current_sentence.append(l)  # Original sentence word
        else:
          # Extract edit information (assuming basic M2 format)
          current_edits.append(l)
      else:
        sentences.append((current_sentence, current_edits))
        current_sentence = []
        current_edits = []
  return sentences


#########################
# Example usage
fname = "./valid.m2"
sentences = read_sentences(fname)

for s in sentences:
  if len(s[0][0])>0:
    t = edit_sentence(s[0][0], s[1])
    print(' '.join(s[0][0].split()[1:]), "\t", t)
