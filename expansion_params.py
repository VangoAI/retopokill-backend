class ExpansionParams:
    def __init__(self, polychord_splits: list[int], num_rotations: int):
        self.polychord_splits = polychord_splits
        self.num_rotations = num_rotations

    def __str__(self):
        return f'ExpansionParams({self.polychord_splits}, {self.num_rotations})'