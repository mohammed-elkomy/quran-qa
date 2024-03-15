'''
The original file was copied from CLEF2020-CheckThat! Task 2: Verified Claim Retrieval, repository
https://github.com/sshaar/clef2020-factchecking-task2/blob/master/lib/format_checker.py
'''

import re
import os
from functools import partial
import argparse
import pandas as pd

from data_scripts import is_tab_sparated

run_columns = ["qid", "Q0", "docid", "rank", "score", "tag"]
RETRIEVED_DOCS_LIMIT = 10
is_float = partial(re.match, r'^-?\d+(?:\.\d+)?$')

LINE_CHECKS = [
    # lambda line: 'Wrong column delimiter' if len(line) == 1 else None,
    lambda line: 'Less columns than expected' if len(line) < len(run_columns) else None,
    lambda line: 'More columns than expected' if len(line) > len(run_columns) else None,
    # lambda line: 'Wrong Q0' if line[run_columns.index('Q0')] != 'Q0' else None,
    lambda line: 'The score is not a float' if not is_float(line[run_columns.index('score')]) else None,
]

def check_retrieved_doc_limit(preditions_file_path, split_token):
    
    df = pd.read_csv(preditions_file_path, sep=split_token, names=run_columns)    
    value_counts = df['qid'].value_counts()
    # Check if any value has rows greater than the upper limit
    values_greater_than_n = value_counts[value_counts > RETRIEVED_DOCS_LIMIT]
    if not values_greater_than_n.empty:
        return values_greater_than_n
    else:
        return True




def is_space_sparated(preditions_file_path):
    with open(preditions_file_path) as tsvfile:
        pair_ids = {}
        for line_no, line_str in enumerate(tsvfile, start=1):
            line = line_str.split(' ')
            if len(line) == 1:
                return False
    return True


def check_format(preditions_file_path):

    space_separted = is_space_sparated(preditions_file_path)
    tab_separted = is_tab_sparated(preditions_file_path)
    if not space_separted and not tab_separted :
        return 'Wrong column delimiter'

    split_token = '\t' if tab_separted else ' ' # split on tab if the file is tab separated, and on space otherwise
    check_limit_answer = check_retrieved_doc_limit(preditions_file_path, split_token)
    if  check_limit_answer is not True:
        error_message = "The number of retrieved passages per query is above the limit. You are only allowed to retrieve up to 10 passages per query. \n \
Ids of questions that exceed the upper limit are: \n" + check_limit_answer.to_string()
        return error_message
    

    with open(preditions_file_path) as tsvfile:
        pair_ids = {}
        for line_no, line_str in enumerate(tsvfile, start=1):
            line = line_str.split(split_token)
            for check in LINE_CHECKS:
                error = check(line)
                if error is not None:
                    return f'{error} on line {line_no} in file: {preditions_file_path}'

            question_id, doc_id = line[run_columns.index('qid')], line[run_columns.index('docid')]
            duplication = pair_ids.get((question_id, doc_id), False)
            if duplication:
                return f'Duplication of pair(question_id={question_id}, doc_id={doc_id}) ' \
                    f'on lines {duplication} and {line_no} in file: {preditions_file_path}'
            else:
                pair_ids[(question_id, doc_id)] = line_no
    return


def check_filename(input_path):

    file_name = os.path.basename(input_path)
    match = re.search(r'^[a-zA-Z0-9]{3,9}[_]{1}[a-zA-Z0-9]{2,9}$', file_name[:-4])
    if not match:
        print(
            f"Error: Your run file name <{file_name}> is incorrect. "
            f"\n\t   Please adopt this naming formt <TeamID_RunID.json> "
            f"\n\t   such that: "
            f"\n\t\t- TeamID can be an alphanumeric with a length between 3 and 9 characters "
            f"\n\t\t- RunID  can be an alphanumeric with a length between 2 and 9 characters "
            f"\n\t\t    For example: bigIR_run01.tsv")
        return False
    
    else:
        return True

 


def check_run(prediction_file):
    
    if check_filename(prediction_file) is False:
        return False
    
    error = check_format(prediction_file)
    if error:
        print(f"Format check: Failed")
        print(f"Reason: {error}")
        return False
    else:
        print(f"Format check: Passed")
        return True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-prediction', '-m', required=True,
                        help='Path to the file containing the model predictions,\
                              which are supposed to be checked')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    check_run(args.model_prediction)