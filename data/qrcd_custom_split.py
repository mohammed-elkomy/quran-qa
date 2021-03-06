from collections import Counter

from datasets import load_dataset, DownloadConfig, concatenate_datasets
from pipe import where
from preprocess import ArabertPreprocessor
from qrcd_read_write import write_to_JSONL_file, PassageQuestion

dataset = load_dataset("qrcd_dataset_loader.py",
                       data_files={'train': 'qrcd/qrcd_v1.1_train.jsonl',
                                   'validation': 'qrcd/qrcd_v1.1_dev.jsonl',
                                   },
                       preprocessor=None
                       )

Counter(dataset["train"]["question"])
Counter(dataset["validation"]["question"])

new_train = ({'ما هي انواع الحيوانات التي ذكرت في القرآن؟': 74,
              'ما هي الدلائل التي تشير بأن الانسان مخير؟': 38,
              'هل هناك إسلام بدون الحديث الشريف؟': 26,
              'ماذا يشمل الإحسان؟': 24,
              'إن كان الله قدّر علي أفعالي فلماذا يحاسبني؟': 27,
              'هل احترم الإسلام الأنبياء؟': 27,
              'ما المخلوقات التي تسبح الله؟': 26,
              'ما هي اسماء المدن المذكورة في القرآن؟': 21,
              'ما هي شروط قبول التوبة؟': 21,
              'هل كرّم الإسلام المرأة؟': 19,
              'من هم قوم موسى؟': 17,
              'على من فُرض الجهاد في سبيل الله؟': 16,
              'ما هي أنواع الجهاد؟': 12,
              'لماذا سيُحاسب ويُعذب الضال يوم القيامة ان كان ""من يضلل الله فما له من هاد"" كما ورد من قوله تعالى في آية 23 و آية 36 من سورة الزمر؟': 9,
              'ما هي الأحداث المتعلقة بداوود عليه السلام؟': 9,
              'هل وجود قراءات مختلفة للقرآن لا يعني تحريفه؟': 9,
              'هل استخدم لفظ (المطر) في القرآن للعذاب فقط؟': 9,
              'هل ذكر القرآن أن التوراة تم تحريفها؟': 8,
              'من هم الأنبياء الذين ذكروا في القرآن على أنهم مسلمون؟': 8,
              'ماذا يشمل البر؟': 8,
              'ما هو أثر الكلام الطيب؟': 8,
              'متى يحل الإسلام دم الشخص؟': 19,
              'هل ورد في القرآن تنبيه من  صوت أو أصوات معيّنة؟': 8,
              'في كم يوم خلق الله الكون؟': 8,
              'من  الذي عايش سيدنا عيسى عليه السلام؟': 7,
              'من هو اخو سيدنا موسى؟': 7,
              'ما هي لغة القرآن؟': 7,
              'هل هناك إشارات في القرآن إلى كروية الأرض؟': 7,
              'ما هي الآيات التي ذكر فيها الشفاء؟': 6,
              'من هم الشفعاء؟': 6,
              'من هو المسيح؟': 5,
              'ما هي الأماكن التي ذُكرت في القرآن كأماكن مقدسة؟': 5,
              'هل أخبر القرآن عن الذرّة؟': 5,
              'هل ورد في القرآن إشارة لصوت ذي تأثير إيجابي على جسم الإنسان؟': 5,
              'من هم أبناء سيدنا ابراهيم عليه السلام؟': 5,
              'من هم الذين عقروا الناقة؟': 5,
              'ما الدليل على أن القرآن صالح لكل مكان وزمان؟': 5,

              'من هم الأبرار؟': 5,
              'من هم الرسل الذين عاشوا في مصر؟': 4,
              'ما هو الكتاب الذي انزل على عيسى؟': 4,
              'من هم الانصار المذكورين في القرآن؟': 4,
              'لماذا جعل الله معجزة سيدنا صالح الناقة؟': 4,
              'من هم قوم شعيب؟': 4,
              'لماذا اهلك الله قوم شعيب؟': 4,
              'هل لم يتغير أي حرف من القرآن؟': 4,
              'هل تحدثت الحيوانات في القرآن؟': 4,
              'هل الضوء هو النور في القرآن؟': 3,
              'هل يجوز سبي النساء واسترقاقهن كما تفعله بعض الجهات المنتسبة للإسلام بالباطل؟': 3,
              'ما هي قصة اصحاب السبت؟': 3,
              'ما هو البر؟': 3,
              'ما هي عقوبة القتل العمد؟': 3,
              'كم عدّة المطلقة؟': 3,
              'من هو ابن زكريا؟': 3,
              'من هم الحواريون؟': 3,
              'هل مات المسيح بالفعل؟': 3,
              'هل يجوز معاملة أهل الكتاب بالبر والحسنى؟': 3,
              'بماذا شبه الله الحياة الدنيا؟': 3,
              'ما هي الإشارات للدماغ أو لأجزاء من الدّماغ في القرآن؟': 3,
              'هل سيجمع الله بين المؤمنين وأبنائهم وأهلهم في الجنة؟': 3,
              'ما الدلائل على أن هناك اختلاف بين الرسول  والنبي؟': 3,
              'هل حجاب المرأة فرض؟': 3,
              'من هو قارون؟': 3,
              'ما هي الشجرة الملعونة؟': 3,
              'ما هي الشجرة التي يأكل منها الكفار في النار؟': 3,
              'كيف اهلك الله قوم عاد؟': 5,
              'لماذا تلبس المرأة المسلمة الحجاب؟': 4,
              'ما هو سبب نزول سيدنا آدم من الجنة؟': 3,
              'كم فترة رضاعة المولود؟': 3,
              'ضد من فُرض الجهاد؟': 8,
              'هل سمح الإسلام بحرية الاعتقاد بالدخول إلى الإسلام؟': 7,
              'من هم القوم الذين حولهم الله إلى قردة؟': 2,
              'لماذا أباح القرآن حرمة الأشهر الحرم؟': 2,
              'ما عقوبة الربا؟': 2,
              'كم ليلة امر الله زكريا الا يكلم الناس؟': 2,
              'هل ذكر القرآن أن الإنجيل تم تحريفه؟': 2,
              'هل حذر القرآن المؤمنين من اتخاذ أهل الكتاب أولياء لهم؟': 2,
              'هل كان سيدنا يوسف عليه السلام رسولا أم نبيا؟': 2,
              'هل عاش سيدنا موسى في مصر؟': 2,
              'من هلك من أهل سيدنا نوح عليه السلام في الطوفان؟': 2,
              'من هو النبي الذى دخل السجن؟': 2,
              'هل  لفظ (العام) مثل لفظ (السّنة) في القرآن؟': 2,
              'من هو النبي الذي علمه الله لغة الطير والحيوان؟': 2,
              'من هو النبي المعروف بالصبر؟': 2,
              'اين عاش قوم عاد؟': 2,
              'هل بعثة النبي محمد تشمل الجن مع الإنس؟': 2,
              'هل تحدث سيدنا محمد مع الجن؟': 2,
              'من الذي صنع عجلا من الحلي لبني إسرائيل؟': 2,
              'هل سيدنا محمد (ص) هو أول المسلمين؟': 2,
              'من هو النبي الذي عايش طالوت؟': 1,
              'لماذا لم يساوِ الإسلام بين الرجل والمرأة في الشهادة أمام القاضي؟': 1,
              'من كفل السيدة مريم؟': 1,
              'من شارك في غزوة بدر؟': 1,
              'ما حكم التعدد في الزواج؟': 1,
              'ما الغاية من الوضوء قبل الصلاة؟': 1,
              'لماذا يتوضأ المسلمون؟': 1,
              'ماذا حدث لقابيل وهابيل؟': 1,
              'ما هي عقوبة السارق؟': 1,
              'ما هي كفارة اليمين؟': 1,
              'مع أن السؤال هو أساس كل العلوم، لماذا نهى الله المؤمنين عن طرح الأسئلة كما جاء في سورة المائدة آية 101؟': 1,
              'ما هي الآيات التي تتحدث عن موضوع الوصية في سورة المائدة؟': 1,
              'هل أشار القرآن الى نقص الأكسجين في المرتفعات؟': 1,
              'كم عدد الاشهر الحرم؟': 1,
              'ما هو الجبل الذي استقرت عليه سفينة نوح؟': 1,
              'كيف خرج سيدنا يوسف عليه السلام من السجن؟': 1,
              'ما هي الأحداث التي وقعت بين سيدنا موسى والخضر؟': 1,
              'ما هي الأحداث المتعلقة بذي القرنين؟': 1,
              'من هو النبي الذي تكلم مع الهدهد؟': 1,
              'من الذي لبث في بطن الحوت؟': 1,
              'ما هي كفارة الظهار؟': 1,
              'من هم المطففون؟': 1,
              'ما هو فضل ليلة القدر؟': 1,
              'ما هي مصارف الزكاة؟': 1,
              'من بنى الكعبة؟': 1,
              'ما هو جزاء من يقول إن لله ولد؟': 1,
              'ما معنى الحطمة؟': 1,
              'هل هناك علاقة بين الخوف أو الاضطرابات النفسية والشيب؟': 1,
              'هل يؤثم الحاكم الذي لا يحكم بما أنزل الله وشرّع؟': 1,
              })

new_dev = {
    'من هم المحسنون؟': 30,
    'لماذا لا يكتفي المسلمون بالقرآن الكريم ويلجأون للسنة أيضاً؟': 25,
    'ما حكم من يرتد عن دين الإسلام؟': 5,
    'ما هو فضل الجهاد في سبيل الله؟': 9,
    'اتهم القرآن بأنه السبب في الدكتاتورية الإسلامية لكونه أباح التكفير وقتال الكفار حتى يسلموا، كيف نرد على ذلك؟': 8,
    'ما هو موقف القرآن من المثلية الجنسية؟': 5,
    'ما هو وصف الحور العين؟': 4,
    'ماذا حدث لسيدنا يونس بعد ان ابتلعه الحوت؟': 3,
    'ما هي عقوبة من يتهم المرأة بالزنا بغير دليل؟': 3,
    'ماهي العلامات والدلائل التي تشير الى موعد يوم القيامة؟': 5,
    'من هو ابو سيدنا يوسف عليه السلام؟': 2,
    'هل هناك إشارات في القرآن عن نهاية الكيان الصهيوني؟': 2,
    'لماذا ألقي سيدنا يوسف عليه السلام في الجب؟': 1,
    'لماذا حرم الله التبني؟': 1,
    'من الذي خسف الله به الأرض؟': 1,
    'ما معنى القارعة؟': 1,
    'هل كلمة (صوم) في القرآن لا تعني صياما عن الأكل والشرب؟': 1,
    'بأي طريقة حث القرآن المؤمنين على مجادلة أهل الكتاب؟': 1,

}

print("old train samples", sum(Counter(dataset["train"]["question"]).values()))
print("old dev sample", sum(Counter(dataset["validation"]["question"]).values()))
print("old train questions", len(Counter(dataset["train"]["question"]).keys()))
print("old dev questions", len(Counter(dataset["validation"]["question"]).keys()))

print("new train samples", sum(Counter(new_train).values()))
print("new dev sample", sum(Counter(new_dev).values()))

print("new train questions", len(Counter(new_train).keys()))
print("new dev questions", len(Counter(new_dev).keys()))

# questions from the dev and train splits have to be disjoint
assert set(Counter(dataset["train"]["question"]).keys()).isdisjoint(set(Counter(dataset["validation"]["question"]).keys()))
assert set(new_dev.keys()).isdisjoint(set(new_train.keys()))

passage_question_objects_train, passage_question_objects_dev = [], []
for sample in concatenate_datasets([dataset["train"], dataset["validation"]]):
    passage_question_dict = {
        "pq_id": sample["id"],
        "passage": sample["context"],
        "surah": int(sample["title"].split(",")[0].replace("surah:", "")),
        "verses": sample["title"].split(",")[1].replace(" verses:", ""),
        "question": sample["question"],
        "answers": [{"text": answer_text, "start_char": answer_start}
                    for answer_text, answer_start
                    in list(zip(sample["answers"]["text"], sample["answers"]["answer_start"]))]
    }

    # instantiate a PassageQuestion object
    pq_object = PassageQuestion(passage_question_dict)

    if sample["question"] in new_train:

        passage_question_objects_train.append(pq_object)
    else:
        assert sample["question"] in new_dev
        passage_question_objects_dev.append(pq_object)

assert len(passage_question_objects_dev) == sum(Counter(new_dev ).values())
assert len(passage_question_objects_train) == sum(Counter(new_train).values())
write_to_JSONL_file(passage_question_objects_train, 'qrcd/qrcd_v1.1_train_my_split.jsonl', include_answers=True)
write_to_JSONL_file(passage_question_objects_dev, 'qrcd/qrcd_v1.1_dev_my_split.jsonl', include_answers=True)
