import torch
import time
import transformers

if __name__ == '__main__':
    t = time.time()
    x = torch.zeros((50273,), dtype=torch.float32).cuda()
    print('took:', time.time() - t)
    print('done')

took: 137.26064133644104
done

took: 0.8175389766693115
done