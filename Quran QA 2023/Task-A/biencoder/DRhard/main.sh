# model_name = "aubmindlab/bert-base-arabertv02"  # @param ["bert-base-arabic-camelbert-ca-78123-arabic_tydiqa", "araelectra-base-discriminator-2703-arabic_tydiqa", "bert-base-arabertv02-440-arabic_tydiqa","------", "aubmindlab/bert-base-arabertv02", "CAMeL-Lab/bert-base-arabic-camelbert-ca", "aubmindlab/araelectra-base-discriminator" ]
# %chdir biencoder/DRhard
# !mkdir "data"
#

!python biencoder/DRhard/star/train.py --do_train \
    --max_query_length 72 \
    --max_doc_length 512 \
    --preprocess_dir ./data/QQA/preprocess \
    --init_path  $model_name \
    --output_dir ./data/QQA/star_train/models \
    --logging_dir ./data/QQA/star_train/log \
    --optimizer_str adamw \
    --learning_rate 1e-6 \
    --gradient_checkpointing --overwrite_output_dir --num_train_epochs $num_train_epochs --save_every_epochs 1 \
    --per_device_train_batch_size 64




!python biencoder/DRhard/star/prepare_hardneg.py \
--data_type "QQA" \
--max_query_length 72 \
--max_doc_length 512 \
--mode train \
--topk 100 \
--max_positives 20 \
--output_inference_dir "evaluate/star"


!python biencoder/DRhard/star/train.py --do_train \
    --max_query_length 72 \
    --max_doc_length 512 \
    --preprocess_dir ./data/QQA/preprocess \
    --init_path  $model_name \
    --output_dir ./data/QQA/star_train/models \
    --logging_dir ./data/QQA/star_train/log \
    --optimizer_str adamw \
    --learning_rate 1e-6 \
    --gradient_checkpointing --overwrite_output_dir --num_train_epochs $num_train_epochs --save_every_epochs 1 \
    --per_device_train_batch_size 64



!python biencoder/DRhard/star/inference.py --data_type QQA --max_doc_length 512 --mode dev --eval_batch_size 256 --do_full_retrieval --topk 1000 --no_tpu --faiss_gpus 0