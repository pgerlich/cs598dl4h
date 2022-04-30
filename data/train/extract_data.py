import os
import re

AST_INPUT_FILE_DIRECTORY = '/Users/paulgerlich/dev/school/cs598dl4h/data/train/beth/ast/'
RECORD_INPUT_FILE_DIRECTORY = '/Users/paulgerlich/dev/school/cs598dl4h/data/train/beth/txt/'



extraction_regex = re.compile('c=\"(.*)\" ([0-9]+):([0-9]+) ([0-9]+):([0-9]+)\|\|.*\|\|a=\"(.*)\"')

"""
Extract the data from the 2010 relations challenge dataset. At the time of writing, available on https://portal.dbmi.hms.harvard.edu/ under the nc2c-nlp datasets.


Records: Each line is a sentence
AST Files: Annotation of assertions c="coronary artery disease" 115:0 115:2||t="problem"||a="present"

Output format: For each line, for each concept within that line, we need to create a new sample as such:

Example (note that I only showed 2 out of the 5 actual concepts for simplicity):

Sentence: Carpal tunnel syndrome , Hypertension , Hyperlipidemia , Arthritis , h/o Bell's Palsy , HOH , s/p Tonsillectomy
Concept + label: carpal tunnel syndrome: present
Concept + label: hypertension: present

Output samples:
<c>Carpal tunnel syndrome</c>, Hypertension , Hyperlipidemia , Arthritis , h/o Bell's Palsy , HOH , s/p Tonsillectomy --- present
Carpal tunnel syndrome, <c>Hypertension</c> , Hyperlipidemia , Arthritis , h/o Bell's Palsy , HOH , s/p Tonsillectomy --- present


Outstanding questions:
What happens to the position markers for the concept during the embedding stage? Is the <c> just a word in the word embedding matrix?

"""
for directories in [
    (
        '/Users/paulgerlich/dev/school/cs598dl4h/data/train/beth/ast/', 
        '/Users/paulgerlich/dev/school/cs598dl4h/data/train/beth/txt/',
        'bet_train.txt',
    ),
    (
        '/Users/paulgerlich/dev/school/cs598dl4h/data/train/partners/ast/', 
        '/Users/paulgerlich/dev/school/cs598dl4h/data/train/partners/txt/',
        'partners_train.txt',
    ),
        (
        '/Users/paulgerlich/dev/school/cs598dl4h/data/test/ast/', 
        '/Users/paulgerlich/dev/school/cs598dl4h/data/test/txt/',
        'test_data.txt',
    )
]:
    ast_dir = directories[0]
    txt_dir = directories[1]
    output_file_dr = directories[2]

    with open(output_file_dr, 'w') as data_file:
        for filename in os.listdir(ast_dir):
            ast_file_path = os.path.join(ast_dir, filename)
            record_file_path = os.path.join(txt_dir, filename.replace('ast', 'txt'))

            # checking if it is a file
            ast_file = open(ast_file_path, 'r')
            record_file_lines = open(record_file_path,'r').read().splitlines()

            for line in ast_file.read().splitlines():
                data = extraction_regex.match(line)

                concept = data.group(1)
                line = int(data.group(2)) - 1
                word_begin = int(data.group(3))
                word_end = int(data.group(5))
                label = data.group(6)

                input_sentence = record_file_lines[line]
                input_text = ' '.join(input_sentence[word_begin:word_end + 1])

                marked_concept = '<c> ' + concept + ' </c>'
                
                data_file.write(input_sentence.replace(concept, marked_concept) + '----' + label + '\n')