# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os

from OCC.Core.gp import gp_Pnt
from OCC.Extend.ShapeFactory import make_edge, make_face, make_wire
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.TopologyUtils import list_of_shapes_to_compound
from tqdm import tqdm

from polygen.data_utils import parse_splits_list


def export_result(vertices, faces):
    
    points = [gp_Pnt(*vertex) for vertex in vertices]

    brep_faces = []
    valid_faces = []
    for face in faces:

        try:
            edges = [make_edge(points[face[i]], points[face[(i+1)%len(face)]]) for i in range(len(face))]
            wire = make_wire(edges)
            brep_face = make_face(wire)

            brep_faces.append(brep_face)
            valid_faces.append(face)

        except:
            continue
        
    brep_shape, is_success = list_of_shapes_to_compound(brep_faces)

    if is_success:
        write_stl_file(brep_shape, os.path.join(args.exp_path, 'face_meshes', f'{name}.stl'))

    with open(os.path.join(args.exp_path, 'pred_faces', f'{name}.json'), 'w') as f:
        json.dump({
            'vertices': vertices,
            'faces': valid_faces,
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, default="lightning_logs/version_X")
    args = parser.parse_args()

    names = parse_splits_list('data/splits/test.txt')
    names = [n.split('.')[0] for n in names]

    os.makedirs(os.path.join(args.exp_path, 'pred_faces'), exist_ok=True)
    os.makedirs(os.path.join(args.exp_path, 'face_meshes'), exist_ok=True)

    for name in tqdm(names):

        with open(os.path.join(args.exp_path, 'results', f'{name}.json'), "r") as f:
            results = json.loads(f.read())

        export_result(results['vertices'], results['faces'])
