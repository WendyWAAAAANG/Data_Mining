# Score function is to calculate the accuracy,precise,recall and f1_score
def score(pred_result, true_result):
    mistakes = 0
    acc_count = 0
    unacc_count = 0
    for w in range(len(pred_result)):
        if pred_result[w] != true_result[w]:
            mistakes += 1
        elif pred_result[w] == 'acc':
            acc_count += 1
        else:
            unacc_count += 1
    # Firstly, regard 'acc' as the positive to calculate 4 indicators
    accuracy = 1 - (mistakes / len(true_result))
    precise_acc = acc_count / pred_result.count('acc')
    recall_acc = acc_count / true_result.count('acc')
    f_measure_acc = 2 * precise_acc * recall_acc / (precise_acc + recall_acc)
    acc_score = {'accuracy': accuracy, 'precise': precise_acc, 'recall': recall_acc, 'f_measure': f_measure_acc}
    print('acc_score:{}'.format(acc_score))

    # Then regard 'unacc' as the positive to calculate 4 indicators
    precise_unacc = unacc_count / pred_result.count('unacc')
    recall_unacc = unacc_count / true_result.count('unacc')
    f_measure_unacc = 2 * precise_unacc * recall_unacc / (precise_unacc + recall_unacc)
    unacc_score = {'accuracy': accuracy, 'precise': precise_unacc, 'recall': recall_unacc, 'f_measure': f_measure_unacc}
    print('unacc_score:{}'.format(unacc_score))
