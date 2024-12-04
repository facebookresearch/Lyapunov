# Copyright (c) 2024-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import numpy as np

from .utils import to_cuda
from .utils import MyTimeoutError
from .envs.ode import last_index


TOLERANCE_THRESHOLD = 1e-1


logger = getLogger()


def idx_to_infix(env, idx, input=True):
    """
    Convert an indexed prefix expression to SymPy.
    """
    prefix = [env.id2word[wid] for wid in idx]
    infix = env.input_to_infix(prefix) if input else env.output_to_infix(prefix)
    return infix


def idx_lyap_output_to_infix(env, idx):
    """
    Convert a Lyapounov output to a list of infix expressions.
    """
    prefix = [env.id2word[wid] for wid in idx]
    prefix = [x.split() for x in " ".join(prefix).split(f" {env.func_separator} ")]
    offset = 1 if env.lyap_predict_stability else 0
    infix = [env.prefix_to_infix(tt)[0] for tt in prefix[offset:]]
    return infix


def check_hypothesis(eq):
    """
    Check a hypothesis for a given equation and its solution.
    """
    env = Evaluator.ENV
    env.rng = np.random.RandomState(0)
    src = [env.id2word[wid] for wid in eq["src"]]
    tgt = [env.id2word[wid] for wid in eq["tgt"]]
    hyp = [env.id2word[wid] for wid in eq["hyp"]]

    try:
        is_valid = env.check_lyap_validity(src, hyp, tgt)
    except MyTimeoutError:
        is_valid = -3
    except Exception as e:
        logger.info(f"Exception: {str(e)}, {hyp}")
        is_valid = -4

    # update hypothesis
    eq["src"] = env.input_to_infix(src)
    eq["tgt"] = tgt
    eq["hyp"] = hyp
    eq["is_valid"] = is_valid
    return eq


class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env

    def run_all_evals(self):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({"epoch": self.trainer.epoch})
        # save statistics about generated data
        if params.export_data:
            scores["total"] = sum(self.trainer.EQUATIONS.values())
            scores["unique"] = len(self.trainer.EQUATIONS)
            scores["unique_prop"] = 100.0 * scores["unique"] / scores["total"]
            return scores

        with torch.no_grad():
            for task in params.tasks:
                eval_tasks = [["valid", 1]]
                for idx in range(2, len(self.trainer.data_path[task])):
                    eval_tasks.append(["test", idx])
                for data_type, data_path_idx in eval_tasks:
                    if params.beam_eval:
                        self.enc_dec_step_beam(data_type, data_path_idx, task, scores, params.eval_size)
                    else:
                        self.enc_dec_step(data_type, data_path_idx, task, scores)

        return scores

    def truncate_at(self, x, xlen):
        pattern = self.env.word2id[self.env.func_separator]
        bs = len(xlen)
        eos = self.env.eos_index
        assert x.shape[1] == bs
        new_seqs = []
        new_lengths = []
        for i in range(bs):
            s = x[: xlen[i], i].tolist()
            assert s[0] == s[-1] == eos
            ns = last_index(s, pattern)
            if ns != len(s):
                s = s[:ns]
                s.append(eos)
            new_seqs.append(s)
            new_lengths.append(len(s))

        # batch sequence
        lengths = torch.LongTensor(new_lengths)
        seqs = torch.LongTensor(lengths.max().item(), bs).fill_(self.env.pad_index)
        for i, s in enumerate(new_seqs):
            seqs[: lengths[i], i].copy_(torch.LongTensor(s))

        return seqs, lengths

    def enc_dec_step(self, data_type, data_path_idx, task, scores):
        """
        Encoding / decoding step.
        """
        params = self.params
        env = self.env
        encoder = self.modules["encoder"].module if params.multi_gpu else self.modules["encoder"]
        decoder = self.modules["decoder"].module if params.multi_gpu else self.modules["decoder"]
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in env.TRAINING_TASKS

        # stats
        xe_loss = 0
        n_valid = torch.zeros(1000, dtype=torch.long)
        n_total = torch.zeros(1000, dtype=torch.long)

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(params.dump_path, f"eval.{data_type}.{task}.{data_path_idx}.{scores['epoch']}")
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        # iterator
        iterator = self.env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            data_path_idx=data_path_idx,
            batch_size=params.batch_size_eval,
            params=params,
            size=None,
        )
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # cuda
            x1, len1, x2, len2 = to_cuda(x1, len1, x2, len2)

            # print status
            if n_total.sum().item() % 500 < params.batch_size_eval:
                logger.info(f"{n_total.sum().item()}/{eval_size}")

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # optionally truncate input
            if params.lyap_drop_last_equation:
                x1_, len1_ = self.truncate_at(x1, len1)
            else:
                x1_, len1_ = x1, len1

            # forward / loss
            encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
            decoded = decoder("fwd", x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1_)
            word_scores, loss = decoder("predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True)

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()

            # export evaluation details
            if params.eval_verbose:
                for i in range(len(len1)):
                    src = idx_to_infix(env, x1[1 : len1[i] - 1, i].tolist(), True)
                    tgt = idx_to_infix(env, x2[1 : len2[i] - 1, i].tolist(), False)
                    s = f"Equation {n_total.sum().item() + i} ({'Valid' if valid[i] else 'Invalid'})\nsrc={src}\ntgt={tgt}\n"
                    if params.eval_verbose_print:
                        logger.info(s)
                    f_export.write(s + "\n")
                    f_export.flush()

            # stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

        # evaluation details
        if params.eval_verbose:
            f_export.close()

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations were evaluated correctly.")

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size
        if data_type == "test":
            data_path_idx_scores = f"{data_path_idx}_"
        else:
            data_path_idx_scores = ""
        scores[f"{data_type}_{task}_{data_path_idx_scores}xe_loss"] = xe_loss / _n_total
        scores[f"{data_type}_{task}_{data_path_idx_scores}acc"] = 100.0 * _n_valid / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            scores[f"{data_type}_{task}_{data_path_idx_scores}acc_{i}"] = 100.0 * n_valid[i].item() / max(n_total[i].item(), 1)

    def enc_dec_step_beam(self, data_type, data_path_idx, task, scores, size=None):
        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        params = self.params
        env = self.env
        encoder = self.modules["encoder"].module if params.multi_gpu else self.modules["encoder"]
        decoder = self.modules["decoder"].module if params.multi_gpu else self.modules["decoder"]
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1, 2]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in env.TRAINING_TASKS
        max_beam_length = params.max_output_len + 2
        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(params.dump_path, f"eval.beam.{data_type}.{task}.{data_path_idx}.{scores['epoch']}")
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        def display_logs(logs, offset):
            """
            Display detailed results about success / fails.
            """
            if params.eval_verbose == 0:
                return
            for i, res in sorted(logs.items()):
                n_valid = sum([int(v) for _, _, v in res["hyps"]])
                s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\nsrc={res['src']}\ntgt={res['tgt']}\n"
                for hyp, score, valid in res["hyps"]:
                    if score is None:
                        s += f"{int(valid)} {hyp}\n"
                    else:
                        s += f"{int(valid)} {score :.3e} {hyp}\n"
                if params.eval_verbose_print:
                    logger.info(s)
                f_export.write(s + "\n")
                f_export.flush()

        # stats
        xe_loss = 0
        n_valid = torch.zeros(1000, params.beam_size, dtype=torch.long)  # beam size en plus
        n_total = torch.zeros(1000, dtype=torch.long)
        n_perfect_match = 0
        n_correct = 0
        n_timeout = 0
        n_optim = 0
        n_input_err = 0
        n_other_err = 0

        # iterator
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            data_path_idx=data_path_idx,
            batch_size=params.batch_size_eval,
            params=params,
            size=size,
        )
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # cuda
            x1, len1, x2, len2 = to_cuda(x1, len1, x2, len2)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # optionally truncate input
            if params.lyap_drop_last_equation:
                x1_, len1_ = self.truncate_at(x1, len1)
            else:
                x1_, len1_ = x1, len1

            bs = len(len1)

            # forward
            encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
            decoded = decoder("fwd", x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1_)
            word_scores, loss = decoder("predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True)

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()
            n_perfect_match += valid.sum().item()

            # save evaluation details
            beam_log = {}
            for i in range(len(len1)):
                src = idx_to_infix(env, x1[1 : len1[i] - 1, i].tolist(), True)
                tgt = idx_to_infix(env, x2[1 : len2[i] - 1, i].tolist(), False)
                if valid[i]:
                    beam_log[i] = {"src": src, "tgt": tgt, "hyps": [(tgt, None, True)]}

            # stats
            xe_loss += loss.item() * len(y)
            n_valid[:, 0].index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

            # continue if everything is correct. if eval_verbose, perform
            # a full beam search, even on correct greedy generations
            if valid.sum() == len(valid) and params.eval_verbose < 2:
                display_logs(beam_log, offset=n_total.sum().item() - bs)
                continue

            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(
                f"({n_total.sum().item()}/{eval_size}) Found {bs - len(invalid_idx)}/{bs} " f"valid top-1 predictions. Generating solutions ..."
            )

            # generate
            _, _, generations = decoder.generate_beam(
                encoded.transpose(0, 1),
                len1_,
                beam_size=params.beam_size,
                length_penalty=params.beam_length_penalty,
                early_stopping=params.beam_early_stopping,
                max_len=max_beam_length,
            )
            # prepare inputs / hypotheses to check
            # if eval_verbose < 2, no beam search on equations solved greedily
            inputs = []
            for i in range(len(generations)):
                if valid[i] and params.eval_verbose < 2:
                    continue
                for j, (score, hyp) in enumerate(sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)):
                    inputs.append(
                        {
                            "i": i,
                            "j": j,
                            "score": score,
                            "src": x1[1 : len1[i] - 1, i].tolist(),
                            "tgt": x2[1 : len2[i] - 1, i].tolist(),
                            "hyp": hyp[1:].tolist(),
                            "task": task,
                        }
                    )

            # check hypotheses with multiprocessing
            outputs = []
            with ProcessPoolExecutor(max_workers=20) as executor:
                for output in executor.map(check_hypothesis, inputs, chunksize=1):
                    outputs.append(output)

            # read results
            for i in range(bs):

                # select hypotheses associated to current equation
                gens = sorted([o for o in outputs if o["i"] == i], key=lambda x: x["j"])
                assert (len(gens) == 0) == (valid[i] and params.eval_verbose < 2) and (i in beam_log) == valid[i]
                if len(gens) == 0:
                    continue

                # source / target
                src = gens[0]["src"]
                tgt = gens[0]["tgt"]
                beam_log[i] = {"src": src, "tgt": tgt, "hyps": []}

                # for each hypothesis
                for j, gen in enumerate(gens):

                    # sanity check
                    assert gen["src"] == src and gen["tgt"] == tgt and gen["i"] == i and gen["j"] == j

                    # if the hypothesis is correct, and we did not find a correct one before
                    is_valid = gen["is_valid"]
                    if is_valid == 1 and not valid[i]:
                        n_valid[nb_ops[i], j] += 1
                        valid[i] = 1

                    # update beam log
                    beam_log[i]["hyps"].append((gen["hyp"], gen["score"], is_valid))
                    if j == 0:
                        n_correct += is_valid != -2
                        n_timeout += is_valid == -3
                        n_optim += is_valid == -1
                        n_input_err += is_valid == -5
                        n_other_err += is_valid == -4

            # valid solutions found with beam search
            logger.info(f"    Found {valid.sum().item()}/{bs} solutions in beam hypotheses.")

            # export evaluation details
            if params.eval_verbose:
                assert len(beam_log) == bs
                display_logs(beam_log, offset=n_total.sum().item() - bs)

        # evaluation details
        if params.eval_verbose:
            f_export.close()
            logger.info(f"Evaluation results written in {eval_path}")

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations were evaluated correctly.")

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size

        if data_type == "test":
            data_path_idx_scores = f"{data_path_idx}_"
        else:
            data_path_idx_scores = ""
        scores[f"{data_type}_{task}_{data_path_idx_scores}xe_loss"] = xe_loss / _n_total
        scores[f"{data_type}_{task}_{data_path_idx_scores}beam_acc"] = 100.0 * _n_valid / _n_total
        scores[f"{data_type}_{task}_{data_path_idx_scores}perfect"] = 100.0 * n_perfect_match / _n_total
        scores[f"{data_type}_{task}_{data_path_idx_scores}correct"] = 100.0 * (n_perfect_match + n_correct) / _n_total

        scores[f"{data_type}_{task}_{data_path_idx_scores}optim"] = 100.0 * n_optim / _n_total
        scores[f"{data_type}_{task}_{data_path_idx_scores}timeout"] = 100.0 * n_timeout / _n_total
        scores[f"{data_type}_{task}_{data_path_idx_scores}input_err"] = 100.0 * n_input_err / _n_total
        scores[f"{data_type}_{task}_{data_path_idx_scores}other_err"] = 100.0 * n_other_err / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            logger.info(f"{i}: {n_valid[i].sum().item()} / {n_total[i].item()} " f"({100. * n_valid[i].sum().item() / max(n_total[i].item(), 1)}%)")
            scores[f"{data_type}_{task}_{data_path_idx_scores}beam_acc_{i}"] = 100.0 * n_valid[i].sum().item() / max(n_total[i].item(), 1)


def convert_to_text(batch, lengths, id2word, params):
    """
    Convert a batch of sequences to a list of text sequences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sequences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(id2word[batch[k, j]])
        sequences.append(" ".join(words))
    return sequences
