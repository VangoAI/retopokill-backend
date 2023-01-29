from patch import Patch
from vertex import Vertex
from expansion_params import ExpansionParams
from utils import get_rotations, rotate
from pulp import LpMaximize, LpProblem, lpSum, LpVariable
from pulp.constants import LpStatusOptimal
import pulp

class Pattern:
    def __init__(self):
        self.vert_graph = []
        self.faces = []
        self.verts = []
        self.side_polychords = []
        self.side_lengths = []
        self.side_verts = []
        self.num_polychords = 0
        self.num_verts = 0

    @staticmethod
    def from_encoding(topology_encoding: str, polychords_encoding: str):
        def decode_topology(encoding):
            CHARS_PER_HEX = 16

            face_graph = [[-1] * 4 for _ in range(len(encoding) // (3 * CHARS_PER_HEX))]
            last_seen = 0
            for i in range(len(face_graph)):
                for j in range(3): # the string only has 3 numbers per face, the first index is implied
                    s = encoding[i * (3 * CHARS_PER_HEX) + j * CHARS_PER_HEX: i * (3 * CHARS_PER_HEX) + j * CHARS_PER_HEX + CHARS_PER_HEX]
                    if s != '#' * CHARS_PER_HEX:
                        face_graph[i][j + 1] = int(s, 16) # turn the hex digits into a decimal
                        if face_graph[i][j + 1] > last_seen:
                            last_seen = face_graph[i][j + 1]
                            face_graph[last_seen][0] = i # set the first index of the new face to the index of the face that points to it
                # get rid of quads that share 2 edges
                d = set()
                for f in face_graph[i]:
                    if f != -1 and f in d:
                        raise Exception('topology has 2 quads that share 2 edges')
                    d.add(f)
            return face_graph

        def decode_polychords(encoding):
            CHARS_PER_HEX = 4
            
            side_polychords = [[]]
            num_polychords = 0
            i = 0
            while i < len(encoding):
                assert encoding[i] == 'x' or encoding[i] == '-'
                if encoding[i] == 'x':
                    polychord_num = int(encoding[i + 1: i + 1 + CHARS_PER_HEX], 16)
                    side_polychords[-1].append(polychord_num)
                    num_polychords = max(num_polychords, polychord_num + 1)
                    i += 5
                elif encoding[i] == '-':
                    side_polychords.append([])
                    i += 1
            return side_polychords, num_polychords

        def get_face_vert_adjacency(face_graph: list[list[int]]):
            '''
            recovers the faces from the face graph
            '''
            def index_backwards(lst: list, val: int):
                for i in range(len(lst) - 1, -1, -1):
                    if lst[i] == val:
                        return i
                return -1

            INDEX_MAP = [(1, 0), (2, 1), (3, 2), (0, 3)]
            
            vert_count = 0
            faces = []
            for i in range(len(face_graph)):
                face = [-1] * 4
                for j in range(4):
                    neighbor = face_graph[i][j]
                    if 0 <= neighbor < i:
                        for k in range(j): # to account for case when 2 quads share 2 sides
                            if face_graph[i][k] == neighbor:
                                reuse_i1, reuse_i2 = INDEX_MAP[index_backwards(face_graph[neighbor], i)]
                                if face[j] != -1 and face[j] != faces[neighbor][reuse_i1] or face[(j + 1) % 4] != -1 and face[(j + 1) % 4] != faces[neighbor][reuse_i2]:
                                    # we chose the wrong one the first time, so we need to swap
                                    face[k], face[(k + 1) % 4] = faces[neighbor][reuse_i1], faces[neighbor][reuse_i2]
                                    reuse_i1, reuse_i2 = INDEX_MAP[face_graph[neighbor].index(i)]
                                break
                        else:
                            reuse_i1, reuse_i2 = INDEX_MAP[face_graph[neighbor].index(i)]
                        face[j], face[(j + 1) % 4] = faces[neighbor][reuse_i1], faces[neighbor][reuse_i2]
                for j in range(4):
                    if face[j] == -1:
                        face[j] = vert_count
                        vert_count += 1
                faces.append(face)
            return faces, vert_count     

        def get_vert_graph(faces: list[list[int]], num_verts: int):
            '''
            recovers the vertex graph, the dual of the face graph
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

        def get_side_verts(face_graph, faces, side_lengths):
            border_vert_map = {} # vert -> vert
            for i in range(len(face_graph)):
                for j in range(4):
                    if face_graph[i][j] == -1:
                        border_vert_map[faces[i][j]] = faces[i][(j + 1) % 4]
            side_lengths = [1] + side_lengths
            sides = []
            curr_vert = 0
            current_side_length = 0
            side_length_index = 0
            while True:
                if current_side_length == side_lengths[side_length_index] - 1: # in terms of edges
                    sides.append([])
                    side_length_index += 1
                    current_side_length = 0
                sides[-1].append((curr_vert, border_vert_map[curr_vert]))
                curr_vert = border_vert_map[curr_vert]
                current_side_length += 1
                if curr_vert == 0:
                    break
            sides = [[edge[0] for edge in side] + [side[-1][1]] for side in sides]
            return sides


        pattern = Pattern()
        face_graph = decode_topology(topology_encoding)
        pattern.side_polychords, pattern.num_polychords = decode_polychords(polychords_encoding)
        pattern.side_lengths = [len(side_polychord) + 1 for side_polychord in pattern.side_polychords] # in terms of num verts
        pattern.faces, pattern.num_verts = get_face_vert_adjacency(face_graph)
        pattern.verts = [Vertex() for _ in range(pattern.num_verts)]
        pattern.vert_graph = get_vert_graph(pattern.faces, pattern.num_verts)
        pattern.side_verts = get_side_verts(face_graph, pattern.faces, pattern.side_lengths)
        return pattern

    def feasible(self, patch: Patch) -> ExpansionParams:
        '''
        if the pattern is feasible for the given patch, return the parameters.
        otherwise, return None.
        '''
        def solve_ilp(patch_side_lengths):
            model = LpProblem(name="check_feasiblility", sense=LpMaximize)
            polychord_variables = []
            for polychord_num in range(self.num_polychords):
                polychord_variables.append(LpVariable(f"polychord_{polychord_num}", lowBound=0, cat='Integer'))
            
            model += lpSum(polychord_variables) # objective function

            for i in range(len(patch_side_lengths)):
                constraint = patch_side_lengths[i] == self.side_lengths[i] + lpSum([polychord_variables[polychord_num] for polychord_num in self.side_polychords[i]])
                model += (constraint, f"side_{i}_length_constraint")
            
            status = model.solve(pulp.get_solver(pulp.listSolvers(onlyAvailable=True)[0], msg=0))
            if status == LpStatusOptimal:
                return [int(polychord_var.value()) for polychord_var in polychord_variables]
            return None

        # make sure the patch and pattern have the same number of sides
        assert len(self.side_lengths) == len(patch.sides)

        patch_side_lengths = [len(side) for side in patch.sides]
        patch_side_lengths_rotations = get_rotations(patch_side_lengths)
        for i, patch_side_lengths_rotation in enumerate(patch_side_lengths_rotations):
            solution = solve_ilp(patch_side_lengths_rotation)
            if solution:
                return ExpansionParams(solution, i)
        return None

    def expand(self, expansion_params: ExpansionParams):
        def split_polychord(pattern: Pattern, polychord_num: int):
            '''
            split the polychord with the given number into the given number of polychords.
            '''
            def get_next_quad(quad_front: list[int], quad_back: list[int]) -> tuple[list[int], list[int]]:
                for i in range(len(pattern.faces)):
                    if quad_front[0] in pattern.faces[i] and quad_front[1] in pattern.faces[i] and quad_back[0] not in pattern.faces[i] and quad_back[1] not in pattern.faces[i]:
                        return [pattern.faces[i][j] for j in range(4) if pattern.faces[i][j] not in curr_quad_front], curr_quad_front
                assert False, "no next quad found"

            def split_quad(quad_front: list[int], quad_back: list[int]):
                # update verts and num_verts
                new_vert = pattern.num_verts
                pattern.num_verts += 1
                pattern.verts.append(Vertex())

                # update faces
                for i in range(len(pattern.faces)):
                    if quad_front[0] in pattern.faces[i] and quad_front[1] in pattern.faces[i] and quad_back[0] in pattern.faces[i] and quad_back[1] in pattern.faces[i]:
                        pattern.faces.pop(i)
                        break
                pattern.faces.append([quad_front[0], [neighbor for neighbor in pattern.vert_graph[quad_front[0]] if neighbor in quad_back][0], new_vert - 1, new_vert]) # preserve CCW vert order
                pattern.faces.append([quad_front[1], [neighbor for neighbor in pattern.vert_graph[quad_front[1]] if neighbor in quad_back][0], new_vert - 1, new_vert])

                # update vert_graph
                pattern.vert_graph.append([quad_front[0], quad_front[1], new_vert - 1])
                pattern.vert_graph[new_vert - 1].append(new_vert)
                pattern.vert_graph[quad_front[0]].append(new_vert)
                pattern.vert_graph[quad_front[1]].append(new_vert)
                pattern.vert_graph[quad_front[0]].remove(quad_front[1])
                pattern.vert_graph[quad_front[1]].remove(quad_front[0])

            start_edge = None
            end_edge = None
            start_edge_i, start_edge_j = -1, -1
            end_edge_i, end_edge_j = -1, -1
            for i in range(len(pattern.side_polychords)):
                for j in range(len(pattern.side_polychords[i])):
                    if pattern.side_polychords[i][j] == polychord_num:
                        if not start_edge:
                            start_edge = pattern.side_verts[i][j], pattern.side_verts[i][j + 1]
                            start_edge_i, start_edge_j = i, j
                        else:
                            end_edge = pattern.side_verts[i][j], pattern.side_verts[i][j + 1]
                            end_edge_i, end_edge_j = i, j
                            break
                if end_edge:
                    break
            assert start_edge and end_edge

            first_new_vert = pattern.num_verts
            pattern.num_verts += 1
            pattern.verts.append(Vertex())
            pattern.vert_graph.append([start_edge[0], start_edge[1]])
            pattern.vert_graph[start_edge[0]].append(first_new_vert)
            pattern.vert_graph[start_edge[1]].append(first_new_vert)
            pattern.vert_graph[start_edge[0]].remove(start_edge[1])
            pattern.vert_graph[start_edge[1]].remove(start_edge[0])

            curr_quad_front, curr_quad_back = [start_edge[0], start_edge[1]], [-1, -1]
            while end_edge[0] not in curr_quad_front and end_edge[1] not in curr_quad_front:
                curr_quad_front, curr_quad_back = get_next_quad(curr_quad_front, curr_quad_back)
                split_quad(curr_quad_front, curr_quad_back)

            # update side_polychords
            pattern.side_polychords[start_edge_i].insert(start_edge_j + 1, pattern.num_polychords)
            pattern.side_polychords[end_edge_i].insert(end_edge_j, pattern.num_polychords)
            pattern.num_polychords += 1

            # update side_verts
            pattern.side_verts[start_edge_i].insert(start_edge_j + 1, first_new_vert)
            pattern.side_verts[end_edge_i].insert(end_edge_j + 1, pattern.num_verts - 1)

            # update side_lengths
            pattern.side_lengths = [len(verts) for verts in pattern.side_verts]

        pattern = self.copy()
        pattern.rotate(-expansion_params.num_rotations)
        for polychord_num, num_splits in enumerate(expansion_params.polychord_splits):
            for _ in range(num_splits):
                split_polychord(pattern, polychord_num)
        return pattern

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
            for side in self.side_verts:
                for i in range(len(side) - 1):
                    total += self.verts[side[i]]
                    count += 1
            avg = total / count

            verts = []
            for i in range(len(self.verts)):
                if any([i in side for side in self.side_verts]):
                    verts.append(self.verts[i])
                else:
                    verts.append(avg.copy())
            return verts

        def laplace_smooth_iter():
            verts = []
            for i in range(len(self.verts)):
                if any([i in side for side in self.side_verts]):
                    verts.append(self.verts[i])
                else:
                    neighbors = self.vert_graph[i]
                    total = Vertex()
                    for j in neighbors:
                        total += self.verts[j] # could make weighted by valence
                    verts.append(total / len(neighbors))
            return verts

        assert(len(self.side_verts) == len(patch.sides))
        assert(all([len(self.side_verts[i]) == len(patch.sides[i]) for i in range(len(self.side_verts))]))

        # set boundary vert positions
        for i in range(len(self.side_verts)):
            for j in range(len(self.side_verts[i])):
                self.verts[self.side_verts[i][j]].set(patch.sides[i][j][0], patch.sides[i][j][1], patch.sides[i][j][2]) # maybe z = 0?

        self.verts = laplace_smooth_direct()

        num_iterations = 10
        for _ in range(num_iterations):
            self.verts = laplace_smooth_iter()
    
    def rotate(self, num_rotations: int):
        self.side_polychords = rotate(self.side_polychords, num_rotations)
        self.side_lengths = rotate(self.side_lengths, num_rotations)
        self.side_verts = rotate(self.side_verts, num_rotations)
        
    def copy(self):
        pattern = Pattern()
        pattern.vert_graph = [neighbors.copy() for neighbors in self.vert_graph]
        pattern.faces = [face.copy() for face in self.faces]
        pattern.side_polychords = [side_polychord.copy() for side_polychord in self.side_polychords]
        pattern.side_lengths = self.side_lengths.copy()
        pattern.side_verts = [side_vert.copy() for side_vert in self.side_verts]
        pattern.num_polychords = self.num_polychords
        pattern.verts = [vert.copy() for vert in self.verts]
        pattern.num_verts = self.num_verts

        return pattern

    def to_json(self):
        verts = [vert.to_json() for vert in self.verts]
        return {'verts': verts, 'faces': self.faces, 'sides': self.side_verts}
