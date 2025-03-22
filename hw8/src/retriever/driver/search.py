import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm

import pytrec_eval


from retriever.searcher import FaissFlatSearcher

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        curr_qid = None
        rank = 0
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                if curr_qid != qid:
                    curr_qid = qid
                    rank = 0
                rank += 1
                f.write(f'{qid} Q0 {idx} {rank} {s} dense\n') # trec formatted

def evaluate(qrel, run, out_file, query_level=False):

    with open(qrel, "r") as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(run, "r") as f_run:
        run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)

    with open(out_file, 'w') as out:

        for query_id, query_measures in sorted(results.items()):
            for measure, value in sorted(query_measures.items()):
                if query_level:
                    out.write("{:25s}{:8s}{:.4f}\n".format(
                        measure, 
                        query_id, 
                        value))

        for measure in sorted(query_measures.keys()):
            out.write("{:25s}{:8s}{:.4f}\n".format(
                measure, 
                "all", 
                pytrec_eval.compute_aggregated_measure(
                    measure, [query_measures[measure] for query_measures in results.values()]
                )))

def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup

def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_metrics_to', required=False)
    parser.add_argument('--qrels', type=str, default=None)
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    retriever = FaissFlatSearcher(p_reps_0)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps)
        look_up += p_lookup

    q_reps, q_lookup = pickle_load(args.query_reps)
    q_reps = q_reps

    retriever.move_index_to_gpu() # need to test with AWS gpus

    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
    logger.info('Index Search Finished')

    logger.info('Saving run')
    write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)

    
    if args.qrels is not None:
        logger.info('Evaluating')
        evaluate(args.qrels, args.save_ranking_to, args.save_metrics_to, query_level=False)
    logger.info('Done')


if __name__ == '__main__':
    main()