"""NG77 loss-only adapter; the original D forward and VJP path is unchanged."""

from ng77_retention import combined_loss


class RetainedObjective:
    def __init__(self, anchors, coefficient):
        self.anchors = anchors
        self.coefficient = coefficient
        self.last = None

    def __call__(self, scores, original):
        total, keep = combined_loss(original, scores, self.anchors, self.coefficient)
        self.last = dict(original_loss=float(original.detach()), keep_loss=float(keep.detach()),
                         retention_coefficient=self.coefficient)
        return total
