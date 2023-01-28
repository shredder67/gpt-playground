import torch

@torch.no_grad()
def eval_train_test_loss(model, train_br, test_br):
    # count of evaluation steps is equal to length of test iter
    eval_steps = test_br.length_before_new_iter 
    performance = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros((eval_steps,))
        br_iter = iter(train_br) if split=='train' else iter(test_br)
        for i in range(eval_steps): # expected to be less then TRAIN_EPOCH_NUM_STEPS
            xb, yb = next(br_iter)
            losses[i] = model(xb, yb)[1].item()
        performance[split] = torch.mean(losses)
    model.train()
    return performance


def train_lm(model, train_blockreader, test_blockreader, learning_rate=1e-3, eval_every_iter=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i, (xb, yb) in enumerate(train_blockreader):
        if i % eval_every_iter == 0:
            losses = eval_train_test_loss(model, train_blockreader, test_blockreader)
            print(f"step #{i} train_loss: {losses['train']:.4f}, test_loss: {losses['test']:.4f}")
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()