from patch import Patch
from vertex import Vertex
from utils import get_rotations, rotate

class Pattern:
    def __init__(self, encoding: str):
        '''
        encoding: a string of hex digits representing the unique counterclockwise bfs traversal of the expanded pattern. 
        the encoding uses hex without the 0x prefix. the special characters '##' represents that this edge is a boundary edge.

        from this encoding, we can recover the pattern.
        self.faces is a list of lists where self.faces[i] is a list of 4 point indices that make up face i.
        self.verts is a list of Vertex objects.

        >>> p = Pattern('0102######0304########0403####')
        >>> p.faces
        [[0, 1, 2, 3], [2, 1, 4, 5], [3, 2, 6, 7], [2, 5, 8, 9], [6, 2, 9, 10]]
        >>> p.sides
        >>> [[0, 1, 4], [4, 5, 8], [8, 9, 10], [10, 6, 7], [7, 3, 0]]
        '''

        def decode():
            CHARS_PER_HEX = 16

            if encoding == '':
                return []
            face_graph = [[-1] * 4 for _ in range(len(encoding) // (3 * CHARS_PER_HEX))]
            last_seen = 0
            for i in range(len(face_graph)): # 6 because 2 chars for a hex number times 3 numbers per face
                for j in range(3): # the string only has 3 numbers per face, the first index is implied
                    s = encoding[i * (3 * CHARS_PER_HEX) + j * CHARS_PER_HEX: i * (3 * CHARS_PER_HEX) + j * CHARS_PER_HEX + CHARS_PER_HEX]
                    if s != '#' * CHARS_PER_HEX:
                        face_graph[i][j + 1] = int(s, 16) # turn the hex digits into a decimal
                        if face_graph[i][j + 1] > last_seen:
                            last_seen = face_graph[i][j + 1]
                            face_graph[last_seen][0] = i # set the first index of the new face to the index of the face that points to it
            return face_graph

        def get_faces(face_graph: list[list[int]]):
            '''
            recovers the faces from the face graph
            '''
            INDEX_MAP = [(1, 0), (2, 1), (3, 2), (0, 3)]
            
            vert_count = 4
            faces = [[0, 1, 2, 3]]
            for i in range(1, len(face_graph)):
                face = []
                neighbor = face_graph[i][0]
                reuse_i1, reuse_i2 = INDEX_MAP[face_graph[neighbor].index(i)]
                face.extend([faces[neighbor][reuse_i1], faces[neighbor][reuse_i2]])

                if 0 <= face_graph[i][1] < i:
                    neighbor = face_graph[i][1]
                    _, reuse_i2 = INDEX_MAP[face_graph[neighbor].index(i)]
                    face.append(faces[neighbor][reuse_i2])

                if 0 <= face_graph[i][2] < i:
                    neighbor = face_graph[i][2]
                    reuse_i1, reuse_i2 = INDEX_MAP[face_graph[neighbor].index(i)]
                    if len(face) == 2:
                        face.extend([reuse_i1, reuse_i2])
                    else:
                        face.append(faces[neighbor][reuse_i2])
                    faces.append(face)
                    continue

                if 0 <= face_graph[i][3] < i:
                    if len(face) == 2:
                        face.append(vert_count)
                        vert_count += 1
                    neighbor = face_graph[i][3]
                    reuse_i1, _ = INDEX_MAP[face_graph[neighbor].index(i)]
                    face.append(faces[neighbor][reuse_i1])
                else:
                    face.append(vert_count)
                    vert_count += 1
                    if len(face) == 3:
                        face.append(vert_count)
                        vert_count += 1
                faces.append(face)
            return faces, vert_count

        def get_vert_graph(faces: list[list[int]], num_verts: int):
            '''
            recovers the vertex graph from faces
            '''
            vert_graph = [[] for _ in range(num_verts)]
            for face in faces:
                for i in range(4):
                    ccw_neighbor = face[(i + 1) % 4]
                    cw_neighbor = face[(i - 1) % 4]
                    if ccw_neighbor not in vert_graph[face[i]]:
                        vert_graph[face[i]].append(ccw_neighbor)
                    if cw_neighbor not in vert_graph[face[i]]:
                        vert_graph[face[i]].append(cw_neighbor)
            return vert_graph

        def get_sides(face_graph: list[list[int]], faces: list[list[int]]):
            '''
            recovers the boundary sides of the pattern from the face graph and faces
            assumes corners vertices must only be part of one face
            '''
            if not face_graph:
                return []

            corners = set()
            for i in range(len(face_graph)):
                for j in range(4):
                    if face_graph[i][j] == -1 and face_graph[i][j - 1] == -1:
                        corners.add(faces[i][j])

            border_vert_map = {} # vert -> vert
            for i in range(len(face_graph)):
                for j in range(4):
                    if face_graph[i][j] == -1:
                        border_vert_map[faces[i][j]] = faces[i][(j + 1) % 4]

            sides = []
            curr_vert = 0
            while True:
                if curr_vert in corners:
                    sides.append([])
                sides[-1].append((curr_vert, border_vert_map[curr_vert]))
                curr_vert = border_vert_map[curr_vert]
                if curr_vert == 0:
                    break
            sides = [[edge[0] for edge in side] + [side[-1][1]] for side in sides]
            return sides
        
        face_graph = decode()
        self.faces, self.num_verts = get_faces(face_graph)
        self.vert_graph = get_vert_graph(self.faces, self.num_verts)
        self.sides = get_sides(face_graph, self.faces)
        self.verts = [Vertex() for _ in range(self.num_verts)]

    def feasible(self, patch: Patch):
        '''
        if the pattern is feasible for the given patch, return the parameters.
        otherwise, return None.
        '''
        # just checks side lengths for now
        if len(self.sides) != len(patch.sides):
            return None

        side_lengths = [len(side) for side in self.sides]
        patch_side_lengths = [len(side) for side in patch.sides]

        rotations = get_rotations(side_lengths)
        for i in range(len(rotations)):
            if rotations[i] == patch_side_lengths:
                return {"rotations": i, 'flip': False}

        side_lengths = side_lengths[::-1] # flip
        rotations = get_rotations(side_lengths)
        for i in range(len(rotations)):
            if rotations[i] == patch_side_lengths:
                return {"rotations": i, 'flip': True}

        # compare the lengths of the sides for all rotations of the pattern
        # for i in range(len(self.sides)):

    def expand(self, params: dict[str, float]):
        '''
        expands the pattern into an expanded pattern using the given parameters.
        '''
        p = Pattern('')
        p.num_verts = self.num_verts
        sides = [s[::-1] for s in self.sides[::-1]] if params["flip"] else self.sides
        p.sides = rotate([side.copy() for side in sides], params["rotations"])
        p.vert_graph = [lst.copy() for lst in self.vert_graph]
        p.faces = [face.copy() for face in self.faces]
        p.verts = [Vertex() for _ in range(p.num_verts)]
        return p

    def fit(self, patch: Patch):
        '''
        fits the pattern vertices to the patch.
        '''
        def laplace_smooth_direct():
            '''
            right now, just sets the inner vertices to all be the average of the boundary vert positions
            '''
            total = Vertex()
            count = 0
            for side in self.sides:
                for i in range(len(side) - 1):
                    total += self.verts[side[i]]
                    count += 1
            avg = total / count

            verts = []
            for i in range(len(self.verts)):
                if any([i in side for side in self.sides]):
                    verts.append(self.verts[i])
                else:
                    verts.append(avg.copy())
            return verts

        def laplace_smooth_iter():
            verts = []
            for i in range(len(self.verts)):
                if any([i in side for side in self.sides]):
                    verts.append(self.verts[i])
                else:
                    neighbors = self.vert_graph[i]
                    total = Vertex()
                    for j in neighbors:
                        total += self.verts[j] # could make weighted by valence
                    verts.append(total / len(neighbors))
            return verts

        assert(len(self.sides) == len(patch.sides))
        assert(all([len(self.sides[i]) == len(patch.sides[i]) for i in range(len(self.sides))]))

        # set boundary vert positions
        for i in range(len(self.sides)):
            for j in range(len(self.sides[i])):
                self.verts[self.sides[i][j]].set(patch.sides[i][j][0], patch.sides[i][j][1], patch.sides[i][j][2]) # maybe z = 0?

        self.verts = laplace_smooth_direct()

        num_iterations = 10
        for _ in range(num_iterations):
            self.verts = laplace_smooth_iter()

    def to_json(self):
        verts = [vert.to_json() for vert in self.verts]
        return {'verts': verts, 'faces': self.faces, 'sides': self.sides}


# p = Pattern('0102######0304########0403####')
# p = Pattern('01##########')
# p = Pattern('0102####03040405######0606070207######080408##05####07')
# patch = Patch([[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], [(3, 0, 0), (3, 1, 0), (3, 2, 0), (3, 3, 0)], [(3, 3, 0), (2, 3, 0), (1, 3, 0), (0, 3, 0)], [(0, 3, 0), (0, 2, 0), (0, 1, 0), (0, 0, 0)]])
# p.fit(patch)
# print(p.faces)
# print(p.verts)
# print(p.sides)