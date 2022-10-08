# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SWAG dataset."""


import csv

import datasets


_CITATION = """\
@inproceedings{zellers2018swagaf,
    title={SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference},
    author={Zellers, Rowan and Bisk, Yonatan and Schwartz, Roy and Choi, Yejin},
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year={2018}
}
"""

_DESCRIPTION = """\
Given a partial description like "she opened the hood of the car,"
humans can reason about the situation and anticipate what might come
next ("then, she examined the engine"). SWAG (Situations With Adversarial Generations)
is a large-scale dataset for this task of grounded commonsense
inference, unifying natural language inference and physically grounded reasoning.
The dataset consists of 113k multiple choice questions about grounded situations
(73k training, 20k validation, 20k test).
Each question is a video caption from LSMDC or ActivityNet Captions,
with four answer choices about what might happen next in the scene.
The correct answer is the (real) video caption for the next event in the video;
the three incorrect answers are adversarially generated and human verified,
so as to fool machines but not humans. SWAG aims to be a benchmark for
evaluating grounded commonsense NLI and for learning representations.
The full data contain more information,
but the regular configuration will be more interesting for modeling
(note that the regular data are shuffled). The test set for leaderboard submission
is under the regular configuration.
"""

_LICENSE = "Unknown"

_URLs = {
    "full": {
        "train": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/train_full.csv",
        "val": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/val_full.csv",
    },
    "regular": {
        "train": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/train.csv",
        "val": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/val.csv",
        "test": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/test.csv",
    },
}


class Swag(datasets.GeneratorBasedBuilder):
    """SWAG dataset"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="regular", description="The configuration to use for modeling."),
        datasets.BuilderConfig(name="sample", description="The sample data."),
    ]

    DEFAULT_CONFIG_NAME = "sample"

    def _info(self):
        features = datasets.Features(
            {
                "video": datasets.Value("string"),
                "frame_count": datasets.Value("string"),
                "width": datasets.Value("string"),
                "height": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answer": datasets.ClassLabel(names=["0", "1", "2", "3", "4"]),
                "qid": datasets.Value("string"),
                "type": datasets.Value("string"),
                "a0": datasets.Value("string"),
                "a1": datasets.Value("string"),
                "a2": datasets.Value("string"),
                "a3": datasets.Value("string"),
                "a4": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://rowanzellers.com/swag/",
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)

        if self.config.name == "regular":
            splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": data_dir["train"],
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": data_dir["val"],
                        "split": "val",
                    },
                ),
            ]
        else:
            splits = [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": data_dir["train-sample"],
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": data_dir["val-sample"],
                        "split": "val",
                    },
                ),
            ]
        if self.config.name == "regular":
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": data_dir["test"], "split": "test"},
                )
            )
        else:
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": data_dir["test-sample"], "split": "test"},
                )
            )

        return splits

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        with open(filepath, "r", encoding="utf-8") as f:
            lines = list(csv.reader(f, delimiter=","))

            for id_, row in enumerate(lines[1:]):
                yield id_, {
                    "video": row[0],
                    "frame_count": row[1],
                    "width": row[2],
                    "height": row[3],
                    "question": row[4],
                    "answer": row[5],
                    "qid": row[6],
                    "type": row[7],
                    "a0": row[8],
                    "a1": row[9],
                    "a2": row[10],
                    "a3": row[11],
                    "a4": row[12],
                }