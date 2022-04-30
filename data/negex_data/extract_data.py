

with open ('/Users/paulgerlich/dev/school/cs598dl4h/data/negex_data/rsAnnotations-1-120-random.txt', 'r') as in_file:
    with open('/Users/paulgerlich/dev/school/cs598dl4h/data/negex_data/negex_data.txt', 'w') as out_file:
        for line in in_file:
            data = line.split('\t')
            
            condition = data[1].lower()
            sentence = data[2].lower()
            negation_status = data[3]

            input_sentence = sentence.replace(condition, '<c> ' + condition + ' </c>')

            if negation_status == 'Negated':
                negation_status = 'abstent'
            elif negation_status == 'Affirmed':
                negation_status == 'present'
            else:
                continue

            if '<c>' in input_sentence:
                out_file.write(input_sentence + '---' + negation_status + '\n')