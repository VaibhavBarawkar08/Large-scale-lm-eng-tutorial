"""
src/custom_data_parallel.py
"""

from torch import nn


# A standard model that outputs logits
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 3)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        return outputs


# A parallel model that outputs loss during the forward pass
class ParallelLossModel(Model):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, labels):
        logits = super(ParallelLossModel, self).forward(inputs)
        loss = nn.CrossEntropyLoss(reduction="mean")(logits, labels)
        return loss
