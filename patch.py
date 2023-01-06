class Patch:
    def __init__(self, sides: list[list[tuple[float, float, float]]]):
        for i in range(len(sides)):
            assert sides[i][0] == sides[(i - 1) % len(sides)][-1]
        self.sides = sides
