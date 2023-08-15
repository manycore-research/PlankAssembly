# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import numpy as np
import trimesh


class PlankModel():
    """ primitive model class """
    def __init__(self, basis, centroid, coeffs):
        self.basis = basis
        self.centroid = centroid
        self.coeffs = coeffs

    @staticmethod
    def from_bounds(bounds):
        basis = np.eye(3)
        centroid = np.mean(bounds, axis=0)
        extends = np.abs(bounds[1] - bounds[0])
        return PlankModel(basis, centroid, extends)

    def build_mesh(self):
        """build mesh"""
        transform = np.identity(4)
        transform[:3, :3] = self.basis
        transform[:3, -1] = self.centroid
        bbox = trimesh.creation.box(self.coeffs, transform)
        return bbox


def build_mesh(planks, transparent=False):
    planks = np.array(planks).flatten().reshape(-1, 6)

    meshes = trimesh.Trimesh()
    for plank in planks[1:]:
        bounds = np.array(plank).reshape(2, 3)
        plank_model = PlankModel.from_bounds(bounds)
        mesh = plank_model.build_mesh()
        meshes += mesh

    if transparent:
        material = trimesh.visual.material.PBRMaterial(
        alphaMode="BLEND", doubleSided=True, alphaCutoff=1.0, baseColorFactor=[1, 1, 1, 0.5])
        texture = trimesh.visual.texture.TextureVisuals(material=material)
        meshes.visual = texture

    return meshes
