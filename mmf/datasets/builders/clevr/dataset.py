import os

import en_vectors_web_lg
import glob
import json
import numpy as np
import re
import torch
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.general import get_mmf_root
from mmf.utils.text import VocabFromText, tokenize

_CONSTANTS = {
    "questions_folder": "questions",
    "dataset_key": "clevr",
    "empty_folder_error": "CLEVR dataset folder is empty.",
    "questions_key": "questions",
    "question_key": "question",
    "answer_key": "answer",
    "train_dataset_key": "train",
    "images_folder": "images",
    "vocabs_folder": "vocabs",
    "features_folder": "feats"
}

_TEMPLATES = {
    "data_folder_missing_error": "Data folder {} for CLEVR is not present.",
    "question_json_file": "CLEVR_{}_questions.json",
    "vocab_file_template": "{}_{}_vocab.txt",
}


class CLEVRDataset(BaseDataset):
    """Dataset for CLEVR. CLEVR is a reasoning task where given an image with some
    3D shapes you have to answer basic questions.

    Args:
        dataset_type (str): type of dataset, train|val|test
        config (DictConfig): Configuration Node representing all of the data necessary
                             to initialize CLEVR dataset class
        data_folder: Root folder in which all of the data will be present if passed
                     replaces default based on data_dir and data_folder in config.

    """

    def __init__(self, config, dataset_type, data_folder=None, *args, **kwargs):
        super().__init__(_CONSTANTS["dataset_key"], config, dataset_type)
        self._data_folder = data_folder
        self._data_dir = os.path.join(get_mmf_root(), config.data_dir)

        if not self._data_folder:
            self._data_folder = os.path.join(self._data_dir, config.data_folder)

        if not os.path.exists(self._data_folder):
            raise RuntimeError(
                _TEMPLATES["data_folder_missing_error"].format(self._data_folder)
            )

        # Check if the folder was actually extracted in the subfolder
        if config.data_folder in os.listdir(self._data_folder):
            self._data_folder = os.path.join(self._data_folder, config.data_folder)

        if len(os.listdir(self._data_folder)) == 0:
            raise FileNotFoundError(_CONSTANTS["empty_folder_error"])

        grid_feat_path_list = []
        image_path = os.path.join(
            self._data_folder, _CONSTANTS["features_folder"], self._dataset_type
        )
        grid_feat_path_list += glob.glob(image_path + '/*.npz')
        self.iid_to_grid_feat_path = self.img_feat_path_load(grid_feat_path_list)

        self.load_questions()

    def load_questions(self):
        # self.image_path = os.path.join(
        #     self._data_folder, _CONSTANTS["images_folder"], self._dataset_type
        # )

        # with open(
        #     os.path.join(
        #         self._data_folder,
        #         _CONSTANTS["questions_folder"],
        #         _TEMPLATES["question_json_file"].format(self._dataset_type),
        #     )
        # ) as f:
        #     self.questions = json.load(f)[_CONSTANTS["questions_key"]]
        #
        #     # Vocab should only be built in main process, as it will repetition of same task
        #     if is_master():
        #         self._build_vocab(self.questions, _CONSTANTS["question_key"])
        #         self._build_vocab(self.questions, _CONSTANTS["answer_key"])
        #     synchronize()
        question_folder = os.path.join(self._data_folder, _CONSTANTS["questions_folder"])
        stat_ques_list = \
            json.load(open(os.path.join(question_folder, 'train'), 'r'))['questions'] + \
            json.load(open(os.path.join(question_folder, 'val'), 'r'))['questions'] + \
            json.load(open(os.path.join(question_folder, 'test'), 'r'))['questions']

        self.token_to_ix, self.pretrained_emb, max_token = self.tokenize(stat_ques_list, use_glove=True)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)
        self.max_token = max_token
        print('Max token length:', max_token, 'Trimmed to:', self.max_token)

        self.ques_list = []
        self.ques_list += json.load(open(os.path.join(question_folder, self._dataset_type, 'r')))['questions']

        stat_ans_list = \
            json.load(open(os.path.join(question_folder, 'train'), 'r'))['questions'] + \
            json.load(open(os.path.join(question_folder, 'val'), 'r'))['questions']
        self.ans_to_ix, self.ix_to_ans = self.ans_stat(stat_ans_list)
        self.ans_size = self.ans_to_ix.__len__()
        print(' ========== Answer token vocab size:', self.ans_size)
        print('Finished!')

    def tokenize(self, stat_ques_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        max_token = 0
        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            if len(words) > max_token:
                max_token = len(words)

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb, max_token

    def __len__(self):
        return len(self.questions)

    def _get_vocab_path(self, attribute):
        return os.path.join(
            self._data_dir,
            _CONSTANTS["vocabs_folder"],
            _TEMPLATES["vocab_file_template"].format(self.dataset_name, attribute),
        )

    def _build_vocab(self, questions, attribute):
        # Vocab should only be built from "train" as val and test are not observed in training
        if self._dataset_type != _CONSTANTS["train_dataset_key"]:
            return

        vocab_file = self._get_vocab_path(attribute)

        # Already exists, no need to recreate
        if os.path.exists(vocab_file):
            return

        # Create necessary dirs if not present
        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)

        sentences = [question[attribute] for question in questions]
        build_attributes = self.config.build_attributes

        # Regex is default one in tokenize i.e. space
        kwargs = {
            "min_count": build_attributes.get("min_count", 1),
            "keep": build_attributes.get("keep", [";", ","]),
            "remove": build_attributes.get("remove", ["?", "."]),
        }

        if attribute == _CONSTANTS["answer_key"]:
            kwargs["only_unk_extra"] = False

        vocab = VocabFromText(sentences, **kwargs)

        with open(vocab_file, "w") as f:
            f.write("\n".join(vocab.word_list))

    def __getitem__(self, idx):
        ques_ix_iter, ans_iter, iid = self.load_ques_ans(idx)

        # Each call to __getitem__ from dataloader returns a Sample class object which
        # collated by our special batch collator to a SampleList which is basically
        # a attribute based batch in layman terms
        current_sample = Sample()
        torch.from_numpy(ques_ix_iter)

        # question = data["question"]
        # tokens = tokenize(question, keep=[";", ","], remove=["?", "."])
        # processed = self.text_processor({"tokens": tokens})
        current_sample.text = torch.from_numpy(ques_ix_iter)
        # current_sample.text_mask = processed["text_mask"]

        # processed = self.answer_processor({"answers": [data["answer"]]})
        # current_sample.answers = processed["answers"]
        current_sample.targets = torch.from_numpy(ans_iter)  # processed["answers_indices"][0]

        # image_path = os.path.join(self.image_path, data["image_filename"])
        # image = Image.open(image_path).convert("RGB")
        # processed = self.image_processor({"image": image})
        current_sample.image = torch.from_numpy(self.load_img_feats(idx, iid))

        return current_sample

    def load_img_feats(self, idx, iid):
        grid_feat = np.load(self.iid_to_grid_feat_path[iid])
        grid_feat_iter = grid_feat['x']

        return grid_feat_iter

    def img_feat_path_load(self, path_list):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = path.split('/')[-1].split('.')[0]
            iid_to_path[iid] = path

        return iid_to_path

    def load_ques_ans(self, idx):
        # if self.__C.RUN_MODE in ['train']:
        ques = self.ques_list[idx]
        iid = str(ques['image_index'])

        # Process question
        ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=self.max_token)
        ans_iter = np.zeros(1)

        # if self.__C.RUN_MODE in ['train']:
        # process answers
        ans = ques['answer']
        ans_iter = self.proc_ans(ans, self.ans_to_ix)

        return ques_ix_iter, ans_iter, iid

    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    def ans_stat(self, stat_ans_list):
        ans_to_ix = {}
        ix_to_ans = {}

        for ans_stat in stat_ans_list:
            ans = ans_stat['answer']

            if ans not in ans_to_ix:
                ix_to_ans[ans_to_ix.__len__()] = ans
                ans_to_ix[ans] = ans_to_ix.__len__()

        return ans_to_ix, ix_to_ans

    def proc_ans(self, ans, ans_to_ix):
        ans_ix = np.zeros(1, np.int64)
        ans_ix[0] = ans_to_ix[ans]

        return ans_ix
