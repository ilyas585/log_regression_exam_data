def TP(df, y_pred='y_pred', y_true='y_true'):
    return len(df[(df[y_pred] == 1) & (df[y_true] == 1)]) / len(df)


def TN(df, y_pred='y_pred', y_true='y_true'):
    return len(df[(df[y_pred] == 0) & (df[y_true] == 0)]) / len(df)


def FP(df, y_pred='y_pred', y_true='y_true'):
    return len(df[(df[y_pred] == 1) & (df[y_true] == 0)]) / len(df)


def FN(df, y_pred='y_pred', y_true='y_true'):
    return len(df[(df[y_pred] == 0) & (df[y_true] == 1)]) / len(df)


def recall(df, pred='y_pred', right='y_true'):
    TP_ = TP(df, pred, right)
    FN_ = FN(df, pred, right)
    try:
        recall = TP_ / (TP_ + FN_)
        return recall
    except:
        return 0


def precision(df, pred='y_pred', right='y_true'):
    TP_ = TP(df, pred, right)
    FP_ = FP(df, pred, right)
    try:
        precision = TP_ / (TP_ + FP_)
        return precision
    except:
        return 0


def f_score(df, pred='y_pred', right='y_true'):
    try:
        precision_ = precision(df, pred, right)
        recall_ = recall(df, pred, right)
        f_score = 2 * precision_ * recall_ / (precision_ + recall_)
        return f_score
    except:
        return 0
