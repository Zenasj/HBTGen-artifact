nums = 2
def compute_loss(predict, target, nums):
    loss_sum = 0
    for num in range(nums):
        pred = int(predict[0][num])
        if target[0][num] in [pred-1, pred, pred+1]:
            loss_sum += 0
        else:
            loss_sum += 1
    return loss_sum