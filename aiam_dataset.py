# coding=utf-8
'''
Reference 1: https://huggingface.co/docs/datasets/dataset_script
Reference 2: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
'''
import json
import os

from PIL import Image

import datasets

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

logger = datasets.logging.get_logger(__name__)

class AIAFunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for AI Asset's FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for AI Asset's FUNSD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(AIAFunsdConfig, self).__init__(**kwargs)

class AIAFunsd(datasets.GeneratorBasedBuilder):
    """AI Asset dataset."""

    BUILDER_CONFIGS = [
        AIAFunsdConfig(name="aiafunsd", version=datasets.Version("1.0.0"), description="AI Asset FUNSD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-PARA", "B-LIST", "B-TABLE"] # O=unknown | B-PARA=para | B-LIST=list | B-TABLE=table
                        )
                    ),
                    "image": datasets.features.Image(),
                }
            ),
            supervised_keys=None,
            # homepage="https://guillaumejaume.github.io/FUNSD/",
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        cwd = os.getcwd()
        dataset_file_path = os.path.join(cwd, "aiam_dataset.zip")

        # downloaded_file = dl_manager.download_and_extract("D:\\Projects\\ML\\HuggingFace\\dataset.zip")
        downloaded_file = dl_manager.download_and_extract(dataset_file_path)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/trainset/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testset/"}
            ),
        ]

    # def get_line_bbox(self, bboxs):
    #     x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    #     y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    #     x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    #     assert x1 >= x0 and y1 >= y0
    #     bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    #     return bbox

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "json")
        img_dir = os.path.join(filepath, "img")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            sFname, sExt = os.path.splitext(file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["annots"]:
                # cur_line_bboxes = []
                words, label, bbox = item["text"], item["type"], item["bbox"]
                # words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                # if label == "other":
                #     for w in words:
                #         tokens.append(w["text"])
                #         ner_tags.append("O")
                #         cur_line_bboxes.append(normalize_bbox(w["box"], size))
                # else:
                #     tokens.append(words[0]["text"])
                #     ner_tags.append("B-" + label.upper())
                #     cur_line_bboxes.append(normalize_bbox(words[0]["box"], size))
                #     for w in words[1:]:
                #         tokens.append(w["text"])
                #         ner_tags.append("I-" + label.upper())
                #         cur_line_bboxes.append(normalize_bbox(w["box"], size))
                # cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                # bboxes.extend(cur_line_bboxes)
                tokens.append(words)
                ner_tags.append(label)
                bboxes.append(normalize_bbox(bbox, size))
            yield guid, {"id": sFname, "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags,
                         "image": image}

###########################################
###########################################
# print("Load dataset")
# exit(0)


# _CITATION = """\
# @article{Jaume2019FUNSDAD,
#   title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
#   author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
#   journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
#   year={2019},
#   volume={2},
#   pages={1-6}
# }
# """

# _DESCRIPTION = """\
# https://guillaumejaume.github.io/FUNSD/
# """


