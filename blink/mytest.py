import argparse
import torch
from tqdm import tqdm
import numpy as np
import json
import pickle
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.biencoder.data_process import (process_mention_data, get_candidate_representation,)
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.crossencoder.train_cross import modify, evaluate
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer


def load_var(load_path):
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    file = open(save_path, 'wb')
    pickle.dump(variable, file)
    file.close()
    print("saved")


def load_test_data():
    data, _ = load_var("models/labeled_Methods.pckl")
    test_data = []
    for i, el in enumerate(data):
        el, _ = el
        context_left = " ".join(el[0])
        mention = el[1]
        context_right = " ".join(el[2])
        id = i
        test_data.append({"id": id, "label": "unknown", "label_id": -1, "context_left": context_left.lower(),
                          "mention": mention.lower(), "context_right": context_right.lower()})
    return test_data


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(samples, tokenizer, biencoder_params["max_context_length"],
                                          biencoder_params["max_cand_length"], silent=True, logger=None,
                                          debug=biencoder_params["debug"],)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"])
    return dataloader


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"])
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
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


def _run_crossencoder(crossencoder, dataloader, context_len, device="cuda"):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(crossencoder, dataloader, device, None, context_len, zeshel=False, silent=False)
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    if accuracy > -1:
        predictions = np.argsort(logits, axis=1)
    else:
        predictions = []

    return accuracy, predictions, logits


def _load_candidates(entity_catalogue, entity_encoding):
    # only load candidate encoding if not using faiss index

    candidate_encoding = torch.load(entity_encoding)
    indexer = None

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
    with open(path, 'rt') as f:
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample['title']
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

        rep = get_candidate_representation(entity_text, tokenizer, max_seq_length, title)
        cand_pool.append(rep["ids"])

    cand_pool = torch.LongTensor(cand_pool)
    return cand_pool


def generate_candidate_pool(tokenizer, params):
    # compute candidate pool from entity list
    entity_desc_list = load_entity_dict(params)
    candidate_pool = get_candidate_pool_tensor(entity_desc_list, tokenizer, params["max_cand_length"])

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

    cand_encode_path = params.get("cand_encode_path", None)

    # candidate encoding is not pre-computed.
    # load/generate candidate pool to compute candidate encoding.
    candidate_pool = generate_candidate_pool(tokenizer, params)

    candidate_encoding = None
    if cand_encode_path is not None:
        # try to load candidate encoding from path
        # if success, avoid computing candidate encoding
        try:
            print("Loading pre-generated candidate encode path.")
            candidate_encoding = torch.load(cand_encode_path)
        except:
            print("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        candidate_encoding = encode_candidate(reranker, candidate_pool, params["encode_batch_size"])

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            print("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)


def encode_candidate(reranker, candidate_pool, encode_batch_size, silent=False):
    reranker.model.eval()
    device = reranker.device
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(candidate_pool, sampler=sampler, batch_size=encode_batch_size)
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


def bi_encoder_step(args, samples, keep_all):
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)

    (candidate_encoding, title2id, id2title,
     id2text, faiss_indexer,) = _load_candidates(args.entity_catalogue, args.entity_encoding)

    dataloader = _process_biencoder_dataloader(samples, biencoder.tokenizer, biencoder_params)

    top_k = args.top_k
    labels, nns, scores = _run_biencoder(biencoder, dataloader, candidate_encoding, top_k, faiss_indexer)

    # biencoder_accuracy = -1
    # recall_at = -1
    # if not keep_all:
    #     # get recall values
    #     top_k = args.top_k
    #     x = []
    #     y = []
    #     for i in range(1, top_k):
    #         temp_y = 0.0
    #         for label, top in zip(labels, nns):
    #             if label in top[:i]:
    #                 temp_y += 1
    #         if len(labels) > 0:
    #             temp_y /= len(labels)
    #         x.append(i)
    #         y.append(temp_y)
    #     # plt.plot(x, y)
    #     biencoder_accuracy = y[0]
    #     recall_at = y[-1]
    #     print("biencoder accuracy: %.4f" % biencoder_accuracy)
    #     print("biencoder recall@%d: %.4f" % (top_k, y[-1]))
    #
    # if args.fast:
    #     predictions = []
    #     for entity_list in nns:
    #         sample_prediction = []
    #         for e_id in entity_list:
    #             e_title = id2title[e_id]
    #             sample_prediction.append(e_title)
    #         predictions.append(sample_prediction)
    #
    #     # use only biencoder
    #     return (biencoder_accuracy, recall_at, -1, -1, len(samples), predictions, scores,)

    return labels, nns, scores, candidate_encoding, title2id, id2title, id2text, faiss_indexer


def cross_encoder_step(args, samples, labels, nns, id2title, id2text, keep_all):
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model

    # load cross encoder model
    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        print("loading crossencoder model")
        with open(args.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["path_to_model"] = args.crossencoder_model
        crossencoder = load_crossencoder(crossencoder_params)

    # prepare crossencoder data
    context_input, candidate_input, label_input = prepare_crossencoder_data(crossencoder.tokenizer, samples, labels,
                                                                            nns, id2title, id2text, keep_all)

    context_input = modify(context_input, candidate_input, crossencoder_params["max_seq_length"])

    dataloader = _process_crossencoder_dataloader(context_input, label_input, crossencoder_params)

    # run crossencoder and get accuracy
    accuracy, index_array, unsorted_scores = _run_crossencoder(crossencoder, dataloader,
                                                               context_len=biencoder_params["max_context_length"], )

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
        print("crossencoder normalized accuracy: %.4f" % crossencoder_normalized_accuracy)

        if len(samples) > 0:
            overall_unormalized_accuracy = (crossencoder_normalized_accuracy * len(label_input) / len(samples))
        print("overall unnormalized accuracy: %.4f" % overall_unormalized_accuracy)
    return len(samples), predictions, scores


def run(args, test_data):
    samples = test_data

    # don't look at labels
    keep_all = (args.interactive or samples[0]["label"] == "unknown" or samples[0]["label_id"] < 0)

    # bi-encoder step
    (labels, nns, scores1, candidate_encoding,
     title2id, id2title, id2text, faiss_indexer) = bi_encoder_step(args, samples, keep_all)

    print("cross encoder step...")
    l, predictions, scores2 = cross_encoder_step(args, samples, labels, nns, id2title, id2text, keep_all)

    return labels, nns, scores1, candidate_encoding, title2id, id2title, id2text, faiss_indexer, predictions, scores2


if __name__ == "__main__":
    models_path = "models/"  # the path where you stored the BLINK models

    config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "top_k": 10,
        "no_cuda": False,
        "encode_batch_size": 4,
        "biencoder_model": models_path + "biencoder_wiki_large.bin",
        "biencoder_config": models_path + "biencoder_wiki_large.json",
        "entity_catalogue": models_path + "new_entity.jsonl",
        "entity_encoding": models_path + "paper_entities.t7",
        "cand_encode_path": models_path + "paper_entities.t7",
        "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
        "crossencoder_config": models_path + "crossencoder_wiki_large.json",
        "fast": False,  # set this to be true if speed is a concern
        "max_cand_length": 128,
        "output_path": "logs/"  # logging directory
    }

    args = argparse.Namespace(**config)

    data_to_link = load_test_data()

    # data_to_link = [{
    #                     "id": 0,
    #                     "label": "unknown",
    #                     "label_id": -1,
    #                     "context_left": "".lower(),
    #                     "mention": "Shakespeare".lower(),
    #                     "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
    #                 },
    #                 {
    #                     "id": 1,
    #                     "label": "unknown",
    #                     "label_id": -1,
    #                     "context_left": "Shakespeare's account of the Roman general".lower(),
    #                     "mention": "Julius Caesar".lower(),
    #                     "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
    #                 }]

    # (candidate_encoding, title2id, id2title, id2text,
    #  wikipedia_id2local_id, faiss_indexer,) = _load_candidates(args.entity_catalogue, args.entity_encoding)

    # params = args.__dict__
    # compute_candidate_encoding(params)

    labels, nns, scores1, candidate_encoding, title2id, id2title, id2text, faiss_indexer, predictions, scores2 = run(args, data_to_link)