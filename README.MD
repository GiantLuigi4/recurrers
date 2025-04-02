# Recurrers

A package for recurrent modeling in pytorch

Recurrers are designed to be easy to work with for arbitrary purposes

For example; say you have a model which every 32 tokens, needs to pause execution to modify the state of previous layers<br>
In pytorch alone, that could be annoying to write the code for<br>
however, with a recurrer, you can setup a compute_grid

### TODO: more instructions or whatever

### Triton
Recurrers has been setup to not degrade performance when compiled with triton

If you make your own recurrer, you may have to also make it so that it properly avoids some of triton's attempted optimizations

Check recurrers.py for a sample recurrer