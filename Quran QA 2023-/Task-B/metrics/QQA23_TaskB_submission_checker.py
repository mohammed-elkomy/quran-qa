'''
July 3, 2023
A script to check a run file for submission to the Qur'an QA 2023 shared Task with respect to:
    - utf-8 encoding
    - json file
    - structure and field names
        {"pq_id":[{"answer":list,"rank":int,"score":float,"strt_token_indx":int,"end_token_indx":int}]}
    - ascending rank order (starting at 1)
    - descending sorted scores
    - No duplicate key values exist
    - No duplicate fields exist
    - Name of the run file
'''
import json, argparse, collections, os, re
import traceback
import logging
import codecs
import string

#global variables
stopWords={'من','الى','إلى','عن','على','في','حتى'}

def _is_punctuation(c):
    exclude = set(string.punctuation)
    exclude.add('،')
    exclude.add('؛')
    exclude.add('؟')
    if c in exclude:
        return True
    return False

def normalize_answer_wAr(s):
    """remove punctuation, some stopwords and extra whitespace."""
    def remove_stopWords(text):
        terms = []
        # must take care of the prefixes before removing stopwords
        for term in text.split():
            if term not in stopWords:
                terms.append(term)
        return " ".join(terms)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc_wAr(text):
        exclude = set(string.punctuation)
        exclude.add('،')
        exclude.add('؛')
        exclude.add('؟')
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_stopWords(remove_punc_wAr(s)))

def normalize_answers_wAr(ss):
    """remove punctuation, some stopwords and extra whitespace."""
    cleaned_ss=[]
    for s in ss:
        s = normalize_answer_wAr(s)
        cleaned_ss.append(s)
    return cleaned_ss

def value_resolver(pairs):
    dct = collections.defaultdict(list)
    for key, value in pairs:
        dct[key].append(value)
    return dct

def load_and_check_duplicates_json(input_path) -> list:
    try:
        output_data = []
        duplicate_cond = True
        with open(input_path, 'r', encoding='utf-8') as f:
            output_data = json.loads(f.read(), object_pairs_hook=value_resolver)

        for key in output_data:
            sub_list = output_data[key]
            if len(sub_list) > 1:
                print(f"Error: Key duplicate(s) detected with value of {key}")
                duplicate_cond=False
            for lst in sub_list:
                for dct in lst:
                    for k in dct:
                        if len(dct[k]) > 1:
                            print(f"Error: Field duplicate(s) detected in {k} field at pq_id = {key}")
                            duplicate_cond=False
        return duplicate_cond
    except:
        return False

def load_json(input_path) -> list:
    try:
        output_data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            output_data = json.loads(f.read())
            return output_data
    except:
        return None
    
def check_utf8(input_path):
    try:
        fh = codecs.open(input_path, 'r', encoding="utf-8") #  Try  windows-1250
        fh.readlines()
        fh.seek(0)
        return True
    except UnicodeDecodeError:
        print("Error: The file is not encoded as UTF-8 correctly.")
        return False
    except :
        print("Fatal Error: File not found. Check if the path is correct")
        return False

def check_structure(data):
    field_cond = True
    for pq_id in data:
        quest_answer_list = data[pq_id]
        tmp_rank_score_list = []
        collect_answers = []

        if len(quest_answer_list)==0: # could be a zero-answer question
            field_cond = True
            continue

        for answer in quest_answer_list:
            try:
                tmp_answer = answer["answer"]
                if len(tmp_answer) == 0:
                    print(f"Warning: You have empty answer at pq_id: {pq_id}")
                collect_answers.append(tmp_answer)
            except KeyError:
                print(f"Error: The answer field name has a problem at pq_id: {pq_id}")
                field_cond = False
                #return field_cond
            try:
                tmp_rank = answer["rank"]
                if isinstance(tmp_rank,float):
                    print(f"Error: The rank field type is a float rather than an integer at pq_id: {pq_id}")
                    field_cond = False
            except KeyError:
                print(f"Error: The rank field name has a problem at pq_id: {pq_id}")
                field_cond = False
                return field_cond
            try:
                tmp_score = answer["score"]
            except KeyError:
                print(f"Error: The score field name has a problem at pq_id: {pq_id}")
                field_cond = False
            try:
                tmp_strt_token_indx = answer["strt_token_indx"]
                if isinstance(tmp_strt_token_indx,float):
                    print(f"Error: The strt_token_indx field type is a float rather than an integer at pq_id: {pq_id}")
                    field_cond = False
            except KeyError:
                print(f"Error: The strt_token_indx field name has a problem at pq_id: {pq_id}")
                field_cond = False
                return field_cond
            try:
                tmp_end_token_indx = answer["end_token_indx"]
                if isinstance(tmp_end_token_indx,float):
                    print(f"Error: The end_token_indx field type is a float rather than an integer at pq_id: {pq_id}")
                    field_cond = False
            except KeyError:
                print(f"Error: The end_token_indx field name has a problem at pq_id: {pq_id}")
                field_cond = False
                return field_cond
            try:
                tmp_strt_token_indx = answer["strt_token_indx"]
                tmp_end_token_indx = answer["end_token_indx"]
                if tmp_end_token_indx < tmp_strt_token_indx:
                    print(f"Error: The end_token_indx should be greater than the strt_token_index at pq_id: {pq_id}")
                    field_cond = False
                    return field_cond
            except KeyError:
                print(f"Error: The strt_token_indx or end_token_indx has a problem at pq_id: {pq_id}")
                field_cond = False
                return field_cond

            try:
                tmp_strt_token_indx = int(answer["strt_token_indx"])
                tmp_end_token_indx = int(answer["end_token_indx"])
                if tmp_end_token_indx-tmp_strt_token_indx + 1 > len(answer["answer"].split()):
                    print(f"Error: The difference between the end_token_indx and strt_token_index exceeds the number of answer tokens at pq_id: {pq_id}")
                    field_cond = False
                    return field_cond
            except KeyError:
                print(f"Error: The strt_token_indx or end_token_indx has a problem at pq_id: {pq_id}")
                field_cond = False
                return field_cond

            try:
                tmp_strt_token_indx = int(answer["strt_token_indx"])
                tmp_end_token_indx = int(answer["end_token_indx"])
                if tmp_end_token_indx-tmp_strt_token_indx + 1 < len(answer["answer"].split()):
                    print(f"Error: The difference between the end_token_indx and strt_token_index is less than the number of answer tokens at pq_id: {pq_id}")
                    field_cond = False
                    return field_cond
            except KeyError:
                print(f"Error: The strt_token_indx or end_token_indx has a problem at pq_id: {pq_id}")
                field_cond = False
                return field_cond


            if "score" in answer.keys() and "rank" in answer.keys():
                tmp_rank_score_list.append([answer["rank"], answer["score"]])

        if len(quest_answer_list) > 10:
            print(f"Warning: Number of answers at {pq_id} exceeds 10")

        current_rank = tmp_rank_score_list[0][0]
        current_score = tmp_rank_score_list[0][1]
        for i in range(1,len(tmp_rank_score_list)):
            assumed_rank = i
            rank = tmp_rank_score_list[i][0]
            score = tmp_rank_score_list[i][1]
            if current_rank!=assumed_rank:
                print(f"Error: The rank of the answer at {pq_id} is incorrect; it should be {assumed_rank} instead of {current_rank}.")
                field_cond = False
            if score < current_score:
                current_score = score
            elif score == current_score:
                print(f"Warning: The answer scores of {pq_id} at rank {current_rank} and rank {rank} are the same.")
            elif score > current_score:
                print(f"Error: the order of answer scores at {pq_id} is not descending.")
                field_cond = False
            if rank > current_rank:
                current_rank = rank
            else:
                print(f"Error: the order of answer ranks at {pq_id} is not ascending.")
                field_cond = False
    return field_cond

def submission_checker(input_path):
    all_clear_cond = True
    # ----------------------------------------------------#
    # Check UTF8
    utf_cond = check_utf8(input_path)
    if utf_cond is False:
        return False
    # ----------------------------------------------------#
    # Check structure, Field Names, length of answers, order of rank
    data = load_json(input_path)
    if data == None:
        print("Fatal Error: File not found OR there is an error with the JSON format.")
        return False

    structure_cond = check_structure(data)
    if structure_cond is False:
        return False

    duplicate_cond  = load_and_check_duplicates_json(input_path)
    if duplicate_cond is False:
        return False
            
    if all_clear_cond:
        print("The run file is correct.")
        return True

def check_filename(input_path):
    fname_cond = True
    file_name = os.path.basename(input_path)

    match = re.search(r'^[a-zA-Z0-9]{3,9}[_]{1}[a-zA-Z0-9]{2,9}$', file_name[:-5])
    if not match:
        fname_cond = False
    return fname_cond, file_name


def check_submission(input_path):
    try:
        if input_path[-5:]!=".json":
            print("Error: The input file is NOT a JSON file!")
            return False
        fname_cond, file_name = check_filename(input_path)

        if not fname_cond:
            print(
                f"Error: Your run file name <{file_name}> is incorrect. "
                f"\n\t   Please adopt this naming formt <TeamID_RunID.json> "
                f"\n\t   such that: "
                f"\n\t\t- TeamID can be an alphanumeric with a length between 3 and 9 characters "
                f"\n\t\t- RunID  can be an alphanumeric with a length between 2 and 9 characters "
                f"\n\t\t    For example: bigIR_run01.json")
            return False

        else:
            return submission_checker(input_path)

    except Exception as e:
        print(traceback.format_exc())
        return False


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='''Submission checker of a run file in JSON format''')
        parser.add_argument('--run_file', required=True, help='Run file should adopt this naming formt <TeamID_RunID.json>')
        args = parser.parse_args()

        input_path = args.run_file
        if input_path[-5:]!=".json":
            print("Error: The input file is NOT a JSON file!")
        else:
            fname_cond, file_name = check_filename(input_path)
            if not fname_cond:
                print(
                    f"Error: Your run file name <{file_name}> is incorrect. "
                    f"\n\t   Please adopt this naming formt <TeamID_RunID.json> "
                    f"\n\t   such that: "
                    f"\n\t\t- TeamID can be an alphanumeric with a length between 3 and 9 characters "
                    f"\n\t\t- RunID  can be an alphanumeric with a length between 2 and 9 characters "
                    f"\n\t\t    For example: bigIR_run01.json")
            else:
                if submission_checker(input_path) is False:
                    print("Please review the above warning(s) or error message(s) related to this run file.")

    except Exception as e:
        print(traceback.format_exc())
        logging.error(traceback.format_exc())
