# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import os

import numpy as np
import shapely
import svgwrite
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Extend.TopologyUtils import (TopologyExplorer, discretize_edge,
                                      list_of_shapes_to_compound)
from shapely.ops import split

O = gp_Pnt(0, 0, 0)
X = gp_Dir(1, 0, 0)
Y = gp_Dir(0, 1, 0)
nY = gp_Dir(0, -1, 0)
Z = gp_Dir(0, 0, 1)

VPS = {
    'f': gp_Ax2(O, nY, X),
    't': gp_Ax2(O, Z, X),
    's': gp_Ax2(O, X, Y),
}


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


def build(bboxes, scale=1280):
    bboxes = np.array(bboxes).reshape(-1, 6)

    bboxes = bboxes / scale

    shapes = []
    for bbox in bboxes[1:]:
        origin = gp_Pnt(*bbox[:3])
        offsets = bbox[3:] - bbox[:3]
        shape = BRepPrimAPI_MakeBox(origin, offsets[0], offsets[1], offsets[2]).Shape()
        shapes.append(shape)
    return list_of_shapes_to_compound(shapes)[0]


def project(shape, view, decimals=3):
    """ Return hidden and visible edges as two lists of edges
    """
    hlr = HLRBRep_Algo()

    hlr.Add(shape)

    projector = HLRAlgo_Projector(VPS[view])

    hlr.Projector(projector)
    hlr.Update()
    hlr.Hide()

    hlr_shapes = HLRBRep_HLRToShape(hlr)

    # get visible and hidden compound
    visible_edges = []
    hidden_edges = []

    visible_compound = hlr_shapes.VCompound()
    if visible_compound:
        visible_edges += list(TopologyExplorer(visible_compound).edges())

    hidden_compound = hlr_shapes.HCompound()
    if hidden_compound:
        hidden_edges += list(TopologyExplorer(hidden_compound).edges())

    # discretize edge
    visible_edges = [get_discretize_edge(edge, decimals) for edge in visible_edges]
    hidden_edges = [get_discretize_edge(edge, decimals) for edge in hidden_edges]

    visible_lines = [shapely.linestrings(edge) for edge in visible_edges]
    hidden_lines = [shapely.linestrings(edge) for edge in hidden_edges]

    # First, split lines by other lines
    lines = [line for line in visible_lines + hidden_lines]
    line_types = [0, ] * len(visible_lines) + [1, ] * len(hidden_lines)

    return lines, line_types


def get_discretize_edge(topods_edge, decimals=3):
    """ Returns a svgwrite.Path for the edge, and the 2d bounding box
    """
    points_3d = discretize_edge(topods_edge)
    points_2d = [[point[0], -point[1]] for point in points_3d]
    points_2d = np.round(points_2d, decimals=decimals).tolist()
    return points_2d


def split_lines_on_crossing_points(lines, types):
    """ Split line on crossing points
    """

    splitted_lines = [lines[0], ]
    splitted_types = [types[0], ]

    for index in range(1, len(lines)):

        splitter = lines[index]
        
        tree = shapely.STRtree(splitted_lines)
        query_indices = tree.query(splitter, predicate='crosses')

        if len(query_indices) > 0:

            for query_index in query_indices:

                query_line = splitted_lines[query_index]
                splitted_line = split(query_line, splitter)

                splitted_lines.extend([line for line in splitted_line.geoms])
                splitted_types.extend([splitted_types[query_index], ] * len(splitted_line.geoms))

                splitted_lines[query_index] = None
                splitted_types[query_index] = None

            splitted_lines = [line for line in splitted_lines if line is not None]
            splitted_types = [line for line in splitted_types if line is not None]

        splitted_lines.append(lines[index])
        splitted_types.append(types[index])

    return splitted_lines, splitted_types


def split_lines_on_endpoints(lines, types):
    """ Split line on endpoints
    """
    
    splitted_lines = []
    splitted_types = []

    endpoints = shapely.multipoints(np.concatenate([np.array(line.coords) for line in lines]))
    endpoints = shapely.get_parts(shapely.extract_unique_points(endpoints)).tolist()

    for line, line_type in zip(lines, types):

        tree = shapely.STRtree(endpoints)
        indices = tree.query(line, predicate='contains')

        if len(indices) > 0:
            splitter = shapely.multipoints([endpoints[index] for index in indices])

            splitted_line = split(line, splitter)

            splitted_lines.extend([line for line in splitted_line.geoms])
            splitted_types.extend([line_type, ] * len(splitted_line.geoms))

        else:
            splitted_lines.append(line)
            splitted_types.append(line_type)

    return splitted_lines, splitted_types


def remove_overlapping_lines(lines, line_types):
    """ remove overlapping lines
    """
    
    # sort lines by line lengths
    lengths = shapely.length(lines)
    indices = np.lexsort((-lengths, line_types))

    # remove degenerated cases
    indices = [index for index in indices if lengths[index] > 0]

    lines = [lines[index] for index in indices]
    line_types = [line_types[index] for index in indices]

    unique_lines = [lines[0], ]
    unique_types = [line_types[0], ]

    for line, line_type in zip(lines[1:], line_types[1:]):

        tree = shapely.STRtree(unique_lines)
        indices = tree.query(line, predicate='covers')

        if len(indices) == 0:
            unique_lines.append(line)
            unique_types.append(line_type)

    return unique_lines, unique_types


def render_svg(lines, line_types, view, name, args):
    dwg = svgwrite.Drawing(os.path.join(args.root, 'data', args.data_type, 'svgs', f'{name}_{view}.svg'))
    dwg.viewbox(-1, -1, 2, 2)
    dwg.defs.add(dwg.style(".vectorEffectClass {vector-effect: non-scaling-stroke;}"))

    for line, line_type in zip(lines, line_types):

        endpoints = shapely.get_coordinates(line)
        svg = svgwrite.shapes.Line(endpoints[0], endpoints[1], fill="none", class_='vectorEffectClass')
        svg.stroke("black", width=args.line_width)

        if line_type == 1:
            svg.dasharray([args.line_width*10, args.line_width*10])

        dwg.add(svg)

    dwg.save(pretty=True)
