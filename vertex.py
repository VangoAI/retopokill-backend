class Vertex:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def set(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def copy(self):
        return Vertex(self.x, self.y, self.z)

    def __add__(self, other: 'Vertex'):
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)

    def __truediv__(self, other: float):
        return Vertex(self.x / other, self.y / other, self.z / other)

    def __repr__(self):
        return f'Vertex({self.x}, {self.y}, {self.z})'