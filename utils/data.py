#%%
import json
import random
from pathlib import Path
from unicodedata import normalize
import argparse


class Data:
    def __init__(self, base_dir="data"):
        self.wikipages_dir = Path(base_dir) / "wiki-pages"
        self.trainjsonl_pth = Path(base_dir) / "train.jsonl"
        self.shared_task_dev_pth = Path(base_dir) / "train_task_dev.jsonl"
        self.shared_task_test_pth = Path(base_dir) / "train_task_test.jsonl"
        self.train_ids = None
        self.wiki_ids = None
        self.wikipages = None
        self.trainjsonl = None

    def get_sub_data(self, train_ids):
        self.train_ids = train_ids
        self.wiki_ids = self.get_wiki_ids(train_ids)

    def get_wikipages(self):
        if self.wikipages is None:
            self.wikipages = self.load_wikipages()
        return self.wikipages.copy()

    def get_processed_trainjsonl(self):
        if self.trainjsonl is None:
            wikipages = self.get_wikipages()
            self.trainjsonl = self.load_processed_trainjsonl(wikipages)
        return self.trainjsonl

    def get_trainjsonl(self):
        return parse_jsonl(self.trainjsonl_pth)

    #%% load wiki-pages
    def load_wikipages(self):
        data = {}
        fnames = list(self.wikipages_dir.glob("wiki-*.jsonl"))
        print("--- Load Wiki-Pages ---")
        for step, fname in enumerate(fnames, 1):
            print(f"[{step}/{len(fnames)}] Processing {fname}")
            wiki_data = parse_jsonl(fname)
            wiki_data_dict = to_dict(wiki_data)
            if self.wiki_ids is not None:
                for wiki_id in self.wiki_ids:
                    if wiki_id in wiki_data_dict:
                        dic = wiki_data_dict[wiki_id]
                        dic["lines"] = dic["lines"].split("\n")
                        data.update({wiki_id: dic})
            else:
                for dic in wiki_data_dict.values():
                    dic["lines"] = dic["lines"].split("\n")
                data.update(wiki_data_dict)
        return data

    #%% load train_jsonl
    def load_processed_trainjsonl(self, wikipages):
        json_list = parse_jsonl(self.trainjsonl_pth)
        out = []
        print("--- Processed train.jsonl ---")
        for dic in json_list:
            if self.train_ids and dic["id"] not in self.train_ids:
                continue
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
            out.append(dic)
        return out

    #%% load train_jsonl
    def get_wiki_ids(self, train_ids: list):
        wiki_ids = set()
        trainjsonl = self.get_trainjsonl()
        for dic in trainjsonl:
            if dic["id"] not in train_ids:
                continue
            for evidence_sets in dic["evidence"]:
                for evidence in evidence_sets:
                    if evidence[2] is None or evidence[3] is None:
                        continue
                    evidence_id = normalize("NFKD", evidence[2])
                    wiki_ids.add(evidence_id)
        return list(wiki_ids)


class SubData(Data):
    def __init__(self, base_dir="data", ratio=0.1, seed=1009):
        super().__init__(base_dir=base_dir)
        trainjsonl = parse_jsonl(Path(base_dir) / "train.jsonl")
        random.seed(seed)
        random.shuffle(trainjsonl)
        train_ids = [
            trainjson["id"] for trainjson in trainjsonl[: int(len(trainjsonl) * ratio)]
        ]
        self.get_sub_data(train_ids)


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


#%% test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processed Data.")
    parser.add_argument(
        "--infolder", type=str, default="data", help="folder that store the data."
    )
    parser.add_argument(
        "--outfolder",
        type=str,
        default="processed_data",
        help="folder to save the processed data.",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.1, help="Ratio of data to get.",
    )
    args = parser.parse_args()

    if args.ratio == 1:
        data = Data(args.infolder)
    else:
        data = SubData(base_dir=args.infolder, ratio=args.ratio)

    wikidata = data.get_wikipages()
    trainjsonl = data.get_processed_trainjsonl()

    Path(args.outfolder).mkdir(parents=True, exist_ok=True)
    with open(
        Path(args.outfolder) / "wikipages.json", "w", encoding="utf-8"
    ) as out_file:
        json.dump(wikidata, out_file, ensure_ascii=False, indent=4)

    with open(Path(args.outfolder) / "train.json", "w", encoding="utf-8") as out_file:
        json.dump(trainjsonl, out_file, ensure_ascii=False, indent=4)
# %%
