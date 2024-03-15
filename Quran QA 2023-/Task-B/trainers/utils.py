import os
import shutil
import subprocess
import time

from transformers.trainer_utils import get_last_checkpoint


def save_to_hub(data_args, model_args, trainer, training_args):
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
    if data_args.dataset is not None:
        kwargs["dataset_tags"] = data_args.dataset
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def save_artifacts_compressed(training_args, data_args):
    zip_name = os.path.split(training_args.my_output_dir)[-1]
    zip_name += "-" + os.path.split(data_args.train_file)[-1].rsplit(".", maxsplit=1)[0].rsplit("_", maxsplit=1)[0]
    shutil.make_archive(zip_name, 'zip', ".",
                        base_dir=os.path.split(training_args.my_output_dir)[-1])
    print("=" * 50)
    print(f"successfully saved {zip_name}.zip")

    # push to drive, so I don't have to download anything from colab myself
    zip_name = f"{zip_name}.zip"
    target_folder = time.strftime("%Y-%m-%d")+"-TASK-B"

    pipe = subprocess.Popen(f"rclone --config ../rclone.conf copy {zip_name} colab4:{target_folder}".split(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    return_code = pipe.wait()

    if return_code == 0:
        print(f"successfully uploaded {zip_name}.zip to drive")
    else:
        print("an error occurred, check the logs")
        print(pipe.stdout.read())
        print(pipe.stderr.read())

    if training_args.save_last_checkpoint_to_drive:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        target_save_path = f"{target_folder}/{zip_name.replace('.zip', '')}"
        pipe = subprocess.Popen(f"rclone --config ../rclone.conf copy {last_checkpoint} colab4:{target_save_path}".split(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        return_code = pipe.wait()

        if return_code == 0:
            print(f"successfully written {last_checkpoint} to drive")
        else:
            print("an error occurred, check the logs")
            print(pipe.stdout.read())
            print(pipe.stderr.read())
