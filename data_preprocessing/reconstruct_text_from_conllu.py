import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import glob
import os


def conllu_to_sentences(conllu_file_path: str) -> list[str]:
    """
    Extracts and returns sentences from a given CoNLL-U file.
    
    :param conllu_file_path: Path to the CoNLL-U file
    :return: A list of sentences in plain text
    """
    sentences = []
    current_sentence = []

    # Open the CoNLL-U file and read line by line
    with open(conllu_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip whitespace to handle any extra spaces or newline characters
            line = line.strip()
            # Check if the line is empty, which indicates the end of a sentence
            if not line:
                if current_sentence:
                    # Join the words of the current sentence into 
                    # a single string
                    sentences.append(' '.join(current_sentence))
                    current_sentence = []  # Reset for the next sentence
                    
            elif line.startswith("#"):  # Skip comment lines
                continue
            else:
                # Split the line into columns (fields are tab-separated)
                columns = line.split('\t')
                if len(columns) > 1:
                    # Append the word to the current sentence
                    current_sentence.append(columns[1])  

        # Add the last sentence if the file doesn't end with a newline
        if current_sentence:
            sentences.append(' '.join(current_sentence))

    return sentences


# Get a list of all CoNLL-U files in the directory
conllu_files = glob.glob('/home/badr/RSC_full/rsc/parsed/*.conllu')

# turned out that not all files have sentences in conllu files
# in that case we need to read raw text instead
txt_files_path = '/home/badr/RSC_full/rsc/raw_ocr_texts/'

# Create the output directory if it doesn't exist
output_dir = 'data/outputs/article_texts_full'
os.makedirs(output_dir, exist_ok=True)

empty_sentences_files = []

# Iterate over each file
for file_path in tqdm(conllu_files):
    # Call the conllu_to_sentences function to extract sentences from the file
    sentences = conllu_to_sentences(file_path)
    
    # Get the file name without the path
    file_name = os.path.basename(file_path)
    
    # Create the output file path
    txt_file_name = file_name.replace('.conllu', '.txt')
    output_file_path = os.path.join(output_dir, txt_file_name)

    if not sentences:
    
        # Try to read the raw text file
        try:
            prefix = "Royal_Society_Corpus_open_v6.0_text_"
            txt_file_name = prefix + txt_file_name

            raw_text_file_path = os.path.join(
                txt_files_path, 
                txt_file_name
            )

            with open(raw_text_file_path, 'r', encoding='utf-8') as text_file:
                # Read the raw text file
                sentences = text_file.read()

            # Write the sentences to the output file
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(sentences)
                              
        except FileNotFoundError:
            #print(f"No sentences found in {file_name}")
            #continue
            empty_sentences_files.append(file_name)

    else:
        # Write the sentences to the output file, one sentence per line
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for sentence in sentences:
                output_file.write(sentence + '\n')

# Print the list of files with empty sentences
print("Files with empty sentences:")
for file_name in empty_sentences_files:
    print(file_name)


# total number of files with empty sentences
print(f"Number of empty files: {len(empty_sentences_files)}")