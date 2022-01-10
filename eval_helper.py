import pickle

import numpy as np
from scipy.sparse import issparse


def zero_division(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 0


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def precision(p, t):
    """
    p, t: two sets of labels/integers
    >>> precision({1, 2, 3, 4}, {1})
    0.25
    """
    return len(t.intersection(p)) / len(p)


def recall(p, t):
    return len(t.intersection(p)) / len(t)


def precision_at_ks(Y_pred_scores, Y_test, ks):
    """
    Y_pred_scores: nd.array of dtype float, entry ij is the score of label j for instance i
    Y_test: list of label ids
    """
    p = []
    r = []
    for k in ks:
        Y_pred = []
        for i in np.arange(Y_pred_scores.shape[0]):
            if issparse(Y_pred_scores):
                idx = np.argsort(Y_pred_scores[i].data)[::-1]
                Y_pred.append(set(Y_pred_scores[i].indices[idx[:k]]))
            else:  # is ndarray
                idx = np.argsort(Y_pred_scores[i, :])[::-1]
                Y_pred.append(set(idx[:k]))

        p.append([precision(yp, set(yt)) for yt, yp in zip(Y_test, Y_pred)])
        r.append([recall(yp, set(yt)) for yt, yp in zip(Y_test, Y_pred)])
    return p, r


def example_based_evaluation(pred, target, threshold, num_example):
    pred = np.greater_equal(pred, threshold).astype(np.int)

    common_label = np.sum(np.multiply(pred, target), axis=1)
    sum_pred = np.sum(pred, axis=1)
    sum_true = np.sum(target, axis=1)

    ebp = np.sum(np.nan_to_num(common_label / sum_pred)) / num_example
    ebr = np.sum(np.nan_to_num(common_label / sum_true)) / num_example
    ebf = 2 * ebp * ebr / (ebp + ebr)

    return (ebp, ebr, ebf)


def micro_macro_eval(pred, target, threshold):
    positive = 1
    negative = 0

    pred = np.greater_equal(pred, threshold).astype(np.int)

    tp = np.logical_and(pred == positive, target == positive).astype(np.int)
    tn = np.logical_and(pred == negative, target == negative).astype(np.int)
    fp = np.logical_and(pred == positive, target == negative).astype(np.int)
    fn = np.logical_and(pred == negative, target == positive).astype(np.int)

    sum_tp = np.sum(tp)
    sum_fp = np.sum(fp)
    sum_fn = np.sum(fn)

    MiP = sum_tp / (sum_tp + sum_fp)
    MiR = sum_tp / (sum_tp + sum_fn)
    MiF = 2 * MiP * MiR / (MiP + MiR)

    MaP = np.average(np.nan_to_num(np.divide(np.sum(tp, axis=0), (np.sum(tp, axis=0) + np.sum(fp, axis=0)))))
    MaR = np.average(np.nan_to_num(np.divide(np.sum(tp, axis=0), (np.sum(tp, axis=0) + np.sum(fn, axis=0)))))
    MaF  = 2 * MaP * MaR / (MaP + MaR)

    return (MiF, MiP, MiR, MaF, MaP, MaR)


def getLabelIndex(labels):
    label_index = np.zeros((len(labels), len(labels[1])))
    for i in range(0, len(labels)):
        index = np.where(labels[i] == 1)
        index = np.asarray(index)
        N = len(labels[1]) - index.size
        index = np.pad(index, [(0, 0), (0, N)], 'constant')
        label_index[i] = index

    label_index = np.array(label_index, dtype=int)
    label_index = label_index.astype(np.int32)
    return label_index


def main():
    P_score = pickle.load(open('../results.pkl', 'rb'))
    P_score = np.concatenate(P_score, axis=0)
    T_score = pickle.load(open('../true.pkl', 'rb'))
    T_score = np.concatenate(T_score, axis=0)
    threshold = np.array([0.000005] * 28470)

    test_labelsIndex = getLabelIndex(T_score)
    precisions = precision_at_ks(P_score, test_labelsIndex, ks=[1, 3, 5, 10, 15])
    print('p@k', sum(precisions[0][0])/len(precisions[0][0]), sum(precisions[0][1])/len(precisions[0][1]),
          sum(precisions[0][2])/len(precisions[0][2]), sum(precisions[0][3])/len(precisions[0][3]), sum(precisions[0][4])/len(precisions[0][4]))
    print('r@k', sum(precisions[1][0])/len(precisions[1][0]), sum(precisions[1][1])/len(precisions[1][1]),
          sum(precisions[1][2])/len(precisions[1][2]), sum(precisions[1][3])/len(precisions[1][3]), sum(precisions[1][4])/len(precisions[1][4]))

    ebp, ebr, ebf = example_based_evaluation(P_score, T_score, threshold)
    print('emp, ebr, ebf', ebp, ebr, ebf)

    MiF, MiP, MiR, MaF, MaP, MaR = micro_macro_eval(P_score, T_score, threshold)
    print('mi/ma', MiF, MiP, MiR, MaF, MaP, MaR)


if __name__ == "__main__":
    main()
