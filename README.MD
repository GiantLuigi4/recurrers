# Recurrers

A package for recurrent modeling in pytorch

Recurrers are designed to be easy to work with for arbitrary purposes

For example; say you have a model which every 32 tokens, needs to pause execution to modify the state of previous layers<br>
In pytorch alone, that could be annoying to write the code for<br>
however, with a recurrer, you can setup a compute_grid

## Usage
```py
model = recurrers.Recurrer([ # this class abstracts basically everything about recurrence away
      # sequence of recurrers, momentum transformer is a recurrer
]).to('cuda')
```
For an example with a single momentum transformer
```py
model = recurrers.Recurrer([
    adapter.FeedNetAdapter(
        nn.Embedding(len(tokenizer.get_vocab()), 640),
    ),
    builtin.MomentumTransformer(640, 0.95),
    adapter.FeedNetAdapter(
        nn.Linear(640, len(tokenizer.get_vocab()))
    )
]).to('cuda')
```

Training is very similar to a standard model:
```py
# grab sample
entry = corpus[i:(i + clen)]
toks = torch.tensor(entry, device='cuda', dtype=torch.long)  # put on gpu
tacky = toks.view(1, -1)  # shape it to (1,S) so that the model doesn't throw an error
state = main.model.make_state()  # only real difference is that you must create a state, ideally every step
out, state = main.model(tacky, state)  # run model
loss = lss(out.view(-1, out.size(-1))[:-1], toks[1:])  # evaluate loss
loss.backward()

# step optimizer
trainer.step()
trainer.zero_grad()
```

## Interoperability
Due to state management, you can't really put a recurrer in the middle of a standard architecture

However, you can put a standard model into the middle of a recurrer

### Triton
Recurrers has been setup to not degrade performance when compiled with triton

If you make your own recurrer, you may have to also make it so that it properly avoids some of triton's attempted optimizations

Check recurrers.py for a sample recurrer