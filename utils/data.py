#%%
import json
from pathlib import Path
from unicodedata import normalize


class Data:
    def __init__(self, base_dir="data"):
        self.wikipages_dir = Path(base_dir) / "wiki-pages"
        self.trainjsonl_pth = Path(base_dir) / "train.jsonl"
        self.shared_task_dev_pth = Path(base_dir) / "train_task_dev.jsonl"
        self.shared_task_test_pth = Path(base_dir) / "train_task_test.jsonl"
        self.wikipages = None
        self.trainjsonl = None

    def get_wikipages(self):
        if self.wikipages is None:
            self.wikipages = self.load_wikipages()
        return self.wikipages.copy()

    def get_trainjsonl(self):
        if self.trainjsonl is None:
            wikipages = self.get_wikipages()
            self.trainjsonl = self.load_trainjsonl(wikipages)
        return self.trainjsonl

    #%% load wiki-pages
    def load_wikipages(self):
        data = {}
        fnames = list(self.wikipages_dir.glob("wiki-*.jsonl"))
        for step, fname in enumerate(fnames, 1):
            print(f"[{step}/{len(fnames)}] Processing {fname}")
            wiki_data = parse_jsonl(fname)
            wiki_data_dict = to_dict(wiki_data)
            for dic in wiki_data_dict.values():
                dic["lines"] = dic["lines"].split("\n")
            data.update(wiki_data_dict)
        return data

    #%% load train_jsonl
    def load_trainjsonl(self, wikipages):
        json_list = parse_jsonl(self.trainjsonl_pth)
        for dic in json_list:
            processed_evidence_sets = []
            for evidence_sets in dic["evidence"]:
                evidence_sentences = []
                for evidence in evidence_sets:
                    if evidence[2] is None or evidence[3] is None:
                        continue
                    evidence_id = normalize("NFKD", evidence[2])
                    sentence_id = evidence[3]
                    evidence_sentence = wikipages[evidence_id]["lines"][sentence_id]
                    evidence_sentences.append(evidence_sentence)
                processed_evidence_sets.append(evidence_sentences)
            dic["evidence_sentences"] = processed_evidence_sets
        return json_list


#%% parse jsonl format data
def parse_jsonl(fname):
    with open(fname, "r", encoding="utf-8") as f:
        json_strs = f.readlines()

    data = []
    for json_str in json_strs:
        json_data = json.loads(json_str)
        data.append(json_data)

    return data


#%% turn list of dictionary (with key id) into a dictionary.
def to_dict(list_of_dict):
    output = {}
    for dic in list_of_dict:
        _id = normalize("NFKD", dic["id"])
        dic.pop("id", None)
        output[_id] = dic

    return output
