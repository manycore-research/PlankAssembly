# from polygen (https://github.com/deepmind/deepmind-research/tree/master/polygen)

import argparse
import json
import os

import networkx as nx
import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt
from OCC.Extend.TopologyUtils import (TopologyExplorer,
                                      list_of_shapes_to_compound)
from tqdm.contrib.concurrent import process_map


def parse_splits_list(splits):
    """ Returns a list of info_file paths
    Args:
        splits (list of strings): each item is a path to a .json file 
            or a path to a .txt file containing a list of paths to .json's.
    """

    if isinstance(splits, str):
        splits = splits.split()
    info_files = []
    for split in splits:
        ext = os.path.splitext(split)[1]
        if ext=='.json':
            info_files.append(split)
        elif ext=='.txt':
            info_files += [info_file.rstrip() for info_file in open(split, 'r')]
        else:
            raise NotImplementedError('%s not a valid info_file type'%split)
    return info_files


def build_shape(bboxes):
    bboxes = np.array(bboxes).reshape(-1, 6)

    shapes = []
    for bbox in bboxes[1:]:
        p1, p2 = gp_Pnt(*bbox[:3]), gp_Pnt(*bbox[3:])
        shape = BRepPrimAPI_MakeBox(p1, p2).Shape()
        shapes.append(shape)
    return list_of_shapes_to_compound(shapes)[0]


def quantize_verts(verts, n_bits=8):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    min_range = -1
    max_range = 1
    range_quantize = 2**n_bits - 1
    verts_quantize = (verts - min_range) * range_quantize / (max_range - min_range)
    return verts_quantize.astype('long')


def face_to_cycles(face):
    """Find cycles in face."""
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g))


def flatten_faces(faces):
    """Converts from list of faces to flat face array with stopping indices."""
    if not faces:
        return np.array([0])
    else:
        l = [f + [-1] for f in faces[:-1]]
        l += [faces[-1] + [-2]]
        return np.array([item for sublist in l for item in sublist]) + 3    # pylint: disable=g-complex-comprehension


def quantize_process_mesh(vertices, faces, tris=None, quantization_bits=8):
    """Quantize vertices, remove resulting duplicates and reindex faces."""
    vertices = quantize_verts(vertices, quantization_bits)
    vertices, inv = np.unique(vertices, axis=0, return_inverse=True)

    # Sort vertices by x then y then z.
    sort_inds = np.lexsort(vertices.T[[2, 1, 0]])
    vertices = vertices[sort_inds]

    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[inv[f]] for f in faces]
    if tris is not None:
        tris = np.array([np.argsort(sort_inds)[inv[t]] for t in tris])

    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
        # Only append faces with more than two verts.
        if c_length > 2:
            d = np.argmin(c)
            # Cyclically permute faces just that first index is the smallest.
            sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    if tris is not None:
        tris = np.array([v for v in tris if len(set(v)) == len(v)])

    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))
    if tris is not None:
        tris = tris.tolist()
        tris.sort(key=lambda f: tuple(sorted(f)))
        tris = np.array(tris)

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
    vertices = vertices[vert_connected]

    # Re-index faces and tris to re-ordered vertices.
    vert_indices = (
        np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
    faces = [vert_indices[f].tolist() for f in faces]
    if tris is not None:
        tris = np.array([vert_indices[t].tolist() for t in tris])

    return vertices, faces, tris


def process_mesh(coords):
    
    shape = build_shape(coords)

    topo = TopologyExplorer(shape)

    vertices = []
    hash_to_index = dict()
    for index, vertex in enumerate(topo.vertices()):
        point = BRep_Tool.Pnt(vertex)
        vertices.append([point.X(), point.Y(), point.Z()])
        hash_to_index[hash(vertex)] = index

    faces = []
    for f in topo.faces():
        face = []
        for w in topo.wires_from_face(f):
            face.append([hash_to_index[hash(v)] for v in topo.ordered_vertices_from_wire(w)])
        faces.extend(face)

    vertices = np.array(vertices)
    faces = np.array(faces)

    # Quantize and sort vertices, remove resulting duplicates, sort and reindex faces.
    vertices, faces, _ = quantize_process_mesh(vertices, faces, quantization_bits=args.quantization_bits)

    # Flatten faces and add 'new face' = 1 and 'stop' = 0 tokens.
    faces = flatten_faces(faces)

    return vertices, faces


def process(name):

    try:
        with open(os.path.join(args.src_path, 'infos', f'{name}.json'), 'r') as f:
            infos = json.loads(f.read())

        vertices, faces = process_mesh(infos['coords'])

        with open(os.path.join(args.tgt_path, 'infos', f'{name}.json'), 'w') as f:
            json.dump({
                **infos,
                'vertices': vertices.tolist(),
                'faces': faces.tolist(),
            }, f, indent=4, separators=(', ', ': '))
    except Exception as msg:
        with open(os.path.join('errors.txt'), 'a') as f:
            f.writelines(f'{name}: {msg}.\n')


def main(args):
    info_files = parse_splits_list([
        os.path.join('data', 'splits', 'train.txt'),
        os.path.join('data', 'splits', 'valid.txt'),
        os.path.join('data', 'splits', 'test.txt')])

    info_files = [info_file.split('.')[0] for info_file in info_files]

    process_map(
        process, info_files,
        max_workers=args.max_workers, chunksize=args.chunksize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', metavar="DIR", default="data/data/complete",
                        help='dataset source path.')
    parser.add_argument('--tgt_path', metavar="DIR", default="data/data/polygen",
                        help='dataset target path.')
    parser.add_argument('--name', type=str, default="",
                        help='data name.')
    parser.add_argument("--max_workers", default=16, type=int,
                        help="maximum number of workers")
    parser.add_argument("--chunksize", default=16, type=int,
                        help="chunk size")
    parser.add_argument("--quantization_bits", default=9, type=int,
                        help="quantization bits")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.tgt_path, 'infos'), exist_ok=True)

    if args.name:
        process(args.name)
    else:
        main(args)
