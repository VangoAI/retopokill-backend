from patch import Patch
from vertex import Vertex

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
            if encoding == '':
                return []
            face_graph = [[-1] * 4 for i in range(len(encoding) // 6)]
            last_seen = 0
            for i in range(len(face_graph)): # 6 because 2 chars for a hex number times 3 numbers per face
                for j in range(3): # the string only has 3 numbers per face, the first index is implied
                    s = encoding[i * 6 + j * 2: i * 6 + j * 2 + 2]
                    if s != '##':
                        face_graph[i][j + 1] = int(s, 16) # turn the 2 digits into a decimal
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

                assert not (0 <= face_graph[i][2] < i)

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
            '''
            total_boundaries = 0
            for face in face_graph:
                for i in range(4):
                    if face[i] == -1:
                        total_boundaries += 1

            # first iteration
            sides = [[faces[0][0]]]
            num_boundaries_seen = 1
            i = 1
            while i < 4 and face_graph[0][i] == -1: # don't infinite loop when face_graph is [[-1, -1, -1, -1]]
                sides[-1].append(faces[0][i])
                sides.append([faces[0][i]])
                num_boundaries_seen += 1
                i += 1

            face_index, prev_face_index = face_graph[0][i], 0
            while num_boundaries_seen < total_boundaries:
                i = (face_graph[face_index].index(prev_face_index) + 1) % 4
                sides[-1].append(faces[face_index][i])
                i = (i + 1) % 4
                num_boundaries_seen += 1
                while face_graph[face_index][i] == -1:
                    sides[-1].append(faces[face_index][i])
                    sides.append([faces[face_index][i]])
                    i = (i + 1) % 4
                    num_boundaries_seen += 1
                face_index, prev_face_index = face_graph[face_index][i], face_index
            return sides[:-1] # hacky fix for extra side bug lol

        
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
        pass

    def expand(self, params: list[float]):
        '''
        expands the pattern into an expanded pattern using the given parameters.
        '''
        pass

    def fit(self, patch: Patch, surface):
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

        def project():
            def project_vert(vert: Vertex):
                pass

            for vert in self.verts:
                # if vert in self.boundary_verts:
                continue

                # idk what this does
                # auto& pn = patch->data(v).laplaceIterative.value;
                # if (pn.tail(3).isZero())
                #     continue;

                # and normalize?
                project_vert(vert)

        assert(len(self.sides) == len(patch.sides))
        assert(all([len(self.sides[i]) == len(patch.sides[i]) for i in range(len(self.sides))]))

        # set boundary vert positions
        for i in range(len(self.sides)):
            for j in range(len(self.sides[i])):
                self.verts[self.sides[i][j]].set(patch.sides[i][j][0], patch.sides[i][j][1], patch.sides[i][j][2]) # maybe z = 0?

        self.verts = laplace_smooth_direct()

        num_iterations = 5
        for _ in range(num_iterations):
            print(self.verts)
            self.verts = laplace_smooth_iter()
            project()


# p = Pattern('0102######0304########0403####')
# p = Pattern('01##########')
# print(p.faces)
# print(p.vert_graph)
# print(p.sides)

p = Pattern('0102####03040405######0606070207######080408##05####07')
patch = Patch([[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], [(3, 0, 0), (3, 1, 0), (3, 2, 0), (3, 3, 0)], [(3, 3, 0), (2, 3, 0), (1, 3, 0), (0, 3, 0)], [(0, 3, 0), (0, 2, 0), (0, 1, 0), (0, 0, 0)]])
print(p.faces)
# print(p.verts)
p.fit(patch, None)
print(p.verts)