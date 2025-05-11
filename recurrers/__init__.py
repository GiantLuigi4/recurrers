import recurrers.recurrer as recurrer

ModelState = recurrer.ModelState
Recurrer = recurrer.Recurrer
RecurrerLayer = recurrer.RecurrerLayer

del recurrer

"""
Various residual feedback layers
"""
import recurrers.residual as residuals

"""
Various helper layers (i.e. knowledge feeders) to try to get more capacity from smaller models
"""
import recurrers.helpers as helpers

"""
Adapters for existing architectures (i.e. GRU/LSTM)
"""
import recurrers.adapter as adapter

"""
Various stateful recurrent models
"""
import recurrers.recurrers as builtin

"""
minimal recurrent neural networks
"""
import recurrers.minrnns as minrnn
