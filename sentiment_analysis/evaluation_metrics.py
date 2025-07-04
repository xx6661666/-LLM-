def compute_accuracy(predictions, labels):
    total = len(labels)
    correct = 0
    fake_correct = 0
    true_correct = 0
    fake_total = sum(1 for l in labels if l == 0)
    true_total = sum(1 for l in labels if l == 1)

    for pred, label in zip(predictions, labels):
        if pred == label:
            correct += 1
            if label == 0:
                fake_correct += 1
            else:
                true_correct += 1

    acc = correct / total
    acc_fake = fake_correct / fake_total if fake_total else 0
    acc_true = true_correct / true_total if true_total else 0

    return acc, acc_fake, acc_true