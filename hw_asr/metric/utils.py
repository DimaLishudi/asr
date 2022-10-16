import editdistance


# 3rd seminar code
def calc_cer(target_text, predicted_text) -> float:
    target_text = target_text.lower()
    predicted_text = predicted_text.lower()
    if len(target_text) == 0:
        return 1
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_text = target_text.lower()
    predicted_text = predicted_text.lower()
    targ_words = target_text.split(' ')
    if len(targ_words) == 0:
        return 1
    pred_words = predicted_text.split(' ')
    return editdistance.eval(targ_words, pred_words) / len(targ_words)
