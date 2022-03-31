import argparse
import os.path
import glob2, pickle

import torch
from tqdm import tqdm
import numpy as np
import json
import pickle
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.crossencoder.train_cross import modify, evaluate
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import blink.candidate_ranking.utils as utils
from blink import build_faiss_index


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


def load_test_data(path):
    data_list = []
    with open(path, "rt") as f:
        i = 0
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample["title"]
            text = sample.get("text", "").strip()
            data_list.append(
                {
                    "id": i,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "".lower(),
                    "mention": title.lower(),
                    "context_right": text.lower(),
                }
            )
            i += 1
    return data_list


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    biencoder.to("cuda")
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        context_input = context_input.to("cuda")
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores


def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda"):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(
        crossencoder,
        dataloader,
        device,
        logger,
        context_len,
        zeshel=False,
        silent=False,
    )
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    if accuracy > -1:
        predictions = np.argsort(logits, axis=1)
    else:
        predictions = []

    return accuracy, predictions, logits


def _load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding)
        indexer = None
    else:
        if logger:
            logger.info("Using faiss index to retrieve entities.")
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexer.deserialize_from(index_path)

    title2id = {}
    id2title = {}
    id2text = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return candidate_encoding, title2id, id2title, id2text, indexer


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


def bi_encoder_step(args, samples, keep_all, logger):
    if logger:
        logger.info("loading biencoder model")
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)

    if logger:
        logger.info("loading candidate entities")
    (candidate_encoding, title2id, id2title, id2text, faiss_indexer) = _load_candidates(
        args.entity_catalogue,
        args.entity_encoding,
        faiss_index=getattr(args, "faiss_index", None),
        index_path=getattr(args, "index_path", None),
        logger=logger,
    )

    if logger:
        logger.info("preparing data for biencoder")
    dataloader = _process_biencoder_dataloader(
        samples, biencoder.tokenizer, biencoder_params
    )

    if logger:
        logger.info("run biencoder")
    top_k = args.top_k
    labels, nns, scores = _run_biencoder(
        biencoder, dataloader, candidate_encoding, top_k, faiss_indexer
    )

    return (
        labels,
        nns,
        scores,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        faiss_indexer,
    )


def cross_encoder_step(args, samples, labels, nns, id2title, id2text, keep_all, logger):
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model

    # load cross encoder model
    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        if logger:
            logger.info("loading crossencoder model")
        with open(args.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["path_to_model"] = args.crossencoder_model
        crossencoder = load_crossencoder(crossencoder_params)

    # prepare crossencoder data
    context_input, candidate_input, label_input = prepare_crossencoder_data(
        crossencoder.tokenizer,
        samples,
        labels,
        nns,
        id2title,
        id2text,
        keep_all,
        crossencoder_params["max_seq_length"],
    )

    context_input = modify(
        context_input, candidate_input, crossencoder_params["max_seq_length"]
    )

    dataloader = _process_crossencoder_dataloader(
        context_input, label_input, crossencoder_params
    )

    # run crossencoder and get accuracy
    accuracy, index_array, unsorted_scores = _run_crossencoder(
        crossencoder,
        dataloader,
        logger,
        context_len=biencoder_params["max_context_length"],
    )

    scores = []
    predictions = []
    for entity_list, index_list, scores_list in zip(nns, index_array, unsorted_scores):

        index_list = index_list.tolist()

        # descending order
        index_list.reverse()

        sample_prediction = []
        sample_scores = []
        for index in index_list:
            e_id = entity_list[index]
            e_title = id2title[e_id]
            sample_prediction.append(e_title)
            sample_scores.append(scores_list[index])
        predictions.append(sample_prediction)
        scores.append(sample_scores)

    crossencoder_normalized_accuracy = -1
    overall_unormalized_accuracy = -1
    if not keep_all:
        crossencoder_normalized_accuracy = accuracy
        print(
            "crossencoder normalized accuracy: %.4f" % crossencoder_normalized_accuracy
        )

        if len(samples) > 0:
            overall_unormalized_accuracy = (
                crossencoder_normalized_accuracy * len(label_input) / len(samples)
            )
        print("overall unnormalized accuracy: %.4f" % overall_unormalized_accuracy)
    return len(samples), predictions, scores


def run(args, test_data, logger):
    samples = test_data
    # don't look at labels
    keep_all = (
        args.interactive
        or samples[0]["label"] == "unknown"
        or samples[0]["label_id"] < 0
    )
    # bi-encoder step
    (
        labels,
        nns,
        bi_encoder_score,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        faiss_indexer,
    ) = bi_encoder_step(args, samples, keep_all, logger)

    print("cross encoder step...")
    l, predictions, cross_encoder_scores = cross_encoder_step(
        args, samples, labels, nns, id2title, id2text, keep_all, logger
    )

    return title2id, id2title, id2text, predictions, cross_encoder_scores


def build_index(output_path, candidate_encoding):
    if os.path.exists(output_path):
        return

    config = {
        "output_path": output_path,
        "candidate_encoding": candidate_encoding,  # file path for candidte encoding.
        "hnsw": False,  # If enabled, use inference time efficient HNSW index
        "save_index": True,  # If enabled, save index
        "index_buffer": 50000,  # Temporal memory data buffer size (in samples) for indexer
    }
    params = argparse.Namespace(**config)
    params = params.__dict__

    build_faiss_index.main(params)


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
                entity_id = (f'{sub_section_index}-{entity_info["key"]}',)
                found_indexes = [
                    i
                    for i in range(len(data_to_link))
                    if data_to_link[i]["id"] == entity_id
                ]
                print("len(found_indexes):", len(found_indexes))
                if len(found_indexes) > 0:
                    found_index = found_indexes[0]
                    print(
                        entity_info["literals"],
                        " -> ",
                        nel_result["predictions"][found_index][0],
                        " -> ",
                        nel_result["scores"][found_index][0],
                    )
                    if nel_result["scores"][found_index][0] > 0:
                        entity_info["entity_id"] = nel_result["predictions"][
                            found_index
                        ][0]
    return triple_file_dict


if __name__ == "__main__":
    models_path = "/mnt/BIG-HDD-STORAGE/ebi/arxiv/processed/Models/models/"  # the path where you stored the BLINK models

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

    base_dir = "/mnt/BIG-HDD-STORAGE/ebi/arxiv/processed"
    triples_dir = f"{base_dir}/triples"
    aug_triples_dir = f"{base_dir}/aug_triples_blink"

    args = argparse.Namespace(**config)

    params = args.__dict__
    compute_candidate_encoding(params)

    # build fiass index from candidate encoded matrix if not exists
    build_index(args.index_path, args.entity_encoding)

    logger = utils.get_logger(args.output_path)

    triple_files = glob2.glob(f"{triples_dir}/*.json")
    pbar = tqdm(triple_files)
    for triple_file in pbar:
        head, tail = os.path.split(triple_file)
        triple_file_dict = json.load(open(triple_file))
        data_to_link = process_triple(triple_file_dict)
        nel_result = run(args, data_to_link, logger)
        aug_triple_file_dict = augment_triple(
            triple_file_dict,
            {"scores": nel_result[4], "predictions": nel_result[3]},
            data_to_link,
        )
        save_dir = os.path.join(aug_triples_dir, tail)
        with open(save_dir, "w") as f_handler:
            json.dump(aug_triple_file_dict, f_handler)
        break

    # data_to_link = load_test_data("models/myKB.jsonl")

    # title2id, id2title, id2text, predictions, cross_encoder_scores = run(args, data_to_link, logger)

    # save_var("find_overlap_results.pckl", [title2id, id2title, id2text, predictions, cross_encoder_scores])
