from qrcd_dataset_loader import PassageQuestion, load_jsonl, dump_jsonl


def read_JSONL_file(file_path, has_answers) -> list:
    data_in_file = load_jsonl(file_path)

    # get list of PassageQuestion objects
    passage_question_objects = []
    for passage_question_dict in data_in_file:
        # instantiate a PassageQuestion object
        pq_object = PassageQuestion(passage_question_dict, has_answers)
        passage_question_objects.append(pq_object)

    print(f"Collected {len(passage_question_objects)} Object from {file_path}")
    return passage_question_objects


def write_to_JSONL_file(passage_question_objects, output_path) -> None:
    # list of dictionaries for the passage_question_objects
    dict_data_list = []
    for pq_object in passage_question_objects:
        dict_data = pq_object.to_dict()
        dict_data_list.append(dict_data)
    dump_jsonl(dict_data_list, output_path)


dev = read_JSONL_file("qrcd/qrcd_v1.1_dev.jsonl", True)
train = read_JSONL_file("qrcd/qrcd_v1.1_train.jsonl", True)

write_to_JSONL_file(train + dev, "qrcd/qrcd_v1.1_train_dev.jsonl")
