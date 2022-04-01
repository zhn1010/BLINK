import argparse
import json
import os
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import random
from tqdm import tqdm
import pickle
import glob2
import blink.main_dense as main_dense
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)


def load_var(load_path):
    file = open(load_path, "rb")
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    file = open(save_path, "wb")
    pickle.dump(variable, file)
    file.close()
    print("saved")


def load_entity_dict(params):
    path = params.get("entity_catalogue", None)
    assert path is not None, "Error! entity_dict_path is empty."

    entity_list = []
    print("Loading entity description from path: " + path)
    with open(path, "rt") as f:
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample["title"]
            text = sample.get("text", "").strip()
            entity_list.append((title, text))

    return entity_list


def get_candidate_pool_tensor(entity_desc_list, tokenizer, max_seq_length):
    print("Convert candidate text to id")
    cand_pool = []
    for entity_desc in tqdm(entity_desc_list):
        if type(entity_desc) is tuple:
            title, entity_text = entity_desc
        else:
            title = None
            entity_text = entity_desc

        rep = get_candidate_representation(
            entity_text, tokenizer, max_seq_length, title
        )
        cand_pool.append(rep["ids"])

    cand_pool = torch.LongTensor(cand_pool)
    return cand_pool


def generate_candidate_pool(tokenizer, params):
    # compute candidate pool from entity list
    entity_desc_list = load_entity_dict(params)
    candidate_pool = get_candidate_pool_tensor(
        entity_desc_list, tokenizer, params["max_cand_length"]
    )

    return candidate_pool


def encode_candidate(reranker, candidate_pool, encode_batch_size, silent=False):
    reranker.model.eval()
    reranker.to("cuda")
    device = "cuda"  # reranker.device
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(
        candidate_pool, sampler=sampler, batch_size=encode_batch_size
    )
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader)

    cand_encode_list = None
    for step, batch in enumerate(iter_):
        cands = batch
        cands = cands.to(device)
        cand_encode = reranker.encode_candidate(cands)
        if cand_encode_list is None:
            cand_encode_list = cand_encode
        else:
            cand_encode_list = torch.cat((cand_encode_list, cand_encode))

    return cand_encode_list


def compute_candidate_encoding(params):
    # Init model
    with open(params["biencoder_config"]) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = params["biencoder_model"]
    reranker = load_biencoder(biencoder_params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    device = reranker.device

    cand_encode_path = params.get("entity_encoding", None)

    if os.path.exists(cand_encode_path):
        return

    # candidate encoding is not pre-computed.
    # load/generate candidate pool to compute candidate encoding.
    candidate_pool = generate_candidate_pool(tokenizer, params)

    candidate_encoding = encode_candidate(
        reranker, candidate_pool, params["encode_batch_size"]
    )

    if cand_encode_path is not None:
        # Save candidate encoding to avoid re-compute
        print("Saving candidate encoding to file " + cand_encode_path)
        torch.save(candidate_encoding, cand_encode_path)


def process_triple(triple_file_dict):
    data_to_link = []
    for sub_section_index, sub_section in enumerate(triple_file_dict):
        if "entities_info" in sub_section:
            for entity_info in sub_section["entities_info"]:
                if entity_info["type"] not in ["Reference", "Code", "Number", "Ost"]:
                    bounds = sorted(
                        entity_info["bounds"],
                        key=lambda item: item[1] - item[0],
                        reverse=True,
                    )
                    proper_bound = None
                    for bound in bounds:
                        if (bound[1] - bound[0]) < 32:
                            proper_bound = bound
                            break
                    if proper_bound:
                        left_context = " ".join(
                            sub_section["tokens"][: proper_bound[0]]
                        )
                        mention = " ".join(
                            sub_section["tokens"][proper_bound[0] : proper_bound[1]]
                        )
                        right_context = " ".join(
                            sub_section["tokens"][proper_bound[1] :]
                        )
                        data_to_link.append(
                            {
                                "id": f'{sub_section_index}-{entity_info["key"]}',
                                "label": "unknown",
                                "label_id": -1,
                                "context_left": left_context.lower(),
                                "mention": mention.lower(),
                                "context_right": right_context.lower(),
                            }
                        )
    return data_to_link


def augment_triple(triple_file_dict, nel_result, data_to_link):
    for sub_section_index, sub_section in enumerate(triple_file_dict):
        if "entities_info" in sub_section:
            for entity_info in sub_section["entities_info"]:
                entity_id = f'{sub_section_index}-{entity_info["key"]}'
                found_indexes = [
                    i
                    for i in range(len(data_to_link))
                    if data_to_link[i]["id"] == entity_id
                ]
                if len(found_indexes) > 0:
                    found_index = found_indexes[0]
                    if nel_result["scores"][found_index][0] > 0:
                        entity_info["entity_id"] = nel_result["predictions"][
                            found_index
                        ][0]
    return triple_file_dict


if __name__ == "__main__":
    models_path = "/mnt/BIG-HDD-STORAGE/ebi/arxiv/processed/Models/models/"  # the path where you stored the BLINK models
    base_dir = "/mnt/BIG-HDD-STORAGE/ebi/arxiv/processed"
    triples_dir = f"{base_dir}/triples"
    aug_triples_dir = f"{base_dir}/aug_triples_blink"

    config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "top_k": 5,
        "no_cuda": False,
        "encode_batch_size": 4,
        "biencoder_model": models_path + "biencoder_wiki_large.bin",
        "biencoder_config": models_path + "biencoder_wiki_large.json",
        "entity_catalogue": models_path + "cs_related_wiki_and_pwc_entities.jsonl",
        "entity_encoding": models_path + "cs_related_wiki_and_pwc_entities.t7",
        "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
        "crossencoder_config": models_path + "crossencoder_wiki_large.json",
        "fast": False,  # set this to be true if speed is a concern
        "max_cand_length": 128,
        "faiss_index": "flat",
        "index_path": models_path
        + "faiss_flat_index_cs_related_wiki_and_pwc_entities.pkl",
        "output_path": "logs/",  # logging directory
    }

    args = argparse.Namespace(**config)

    params = args.__dict__
    compute_candidate_encoding(params)

    models = main_dense.load_models(args, logger=None)

    triple_files = glob2.glob(f"{triples_dir}/*.json")
    random.shuffle(triple_files)
    pbar = tqdm(triple_files)
    for triple_file in pbar:
        head, tail = os.path.split(triple_file)
        save_dir = os.path.join(aug_triples_dir, tail)
        if not os.path.exists(save_dir):
            triple_file_dict = json.load(open(triple_file))
            data_to_link = process_triple(triple_file_dict)
            (
                _,
                _,
                _,
                _,
                _,
                predictions,
                scores,
            ) = main_dense.run(args, None, *models, test_data=data_to_link)
            aug_triple_file_dict = augment_triple(
                triple_file_dict,
                {"scores": scores, "predictions": predictions},
                data_to_link,
            )

            with open(save_dir, "w") as f_handler:
                json.dump(aug_triple_file_dict, f_handler, indent=4)
