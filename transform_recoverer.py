#!/usr/bin/env python3
"""
Tool that recovers the transforms that were applied to a base "referencemesh"
from copies of that mesh. E.g. the result of Michael Fogleman's `pack3d`.
"""

import argparse
import logging

import numpy as np
import trimesh as tm

logging.basicConfig(level=logging.INFO)
np.random.seed(123)
np.set_printoptions(suppress=True, precision=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "referencemesh", help="Path to the mesh file for the reference object"
    )
    parser.add_argument(
        "groupmesh",
        help="Path to the mesh that contains transformed instances of `referencemesh`",
    )
    parser.add_argument(
        "--outcsv",
        "-o",
        help="Write resulting 4x4 transformation matrices (flattened, row-major) to file.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Whether to verify results by transforming reference mesh and checking resulting verts.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--debug", action="store_true")
    group.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Set logging level to debug")
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    logging.info("Loading meshes...")
    referencemesh = tm.load(args.referencemesh, process=False)
    groupmesh = tm.load(args.groupmesh, process=False)

    logging.info("Sanitychecking meshes...")
    if not meshes_have_same_metric(groupmesh, referencemesh):
        logging.error(
            "groupmesh doesnt seem to contain copies of referencemesh. Intra-mesh vertex distances are different!"
        )
        exit(1)

    num_verts_ref = len(referencemesh.vertices)
    logging.debug("Reference has %d vertices", num_verts_ref)

    # To save on execution time, use 100 verts at most
    num_corresps = min(num_verts_ref, 100)
    corresponding_idxs = np.random.choice(np.arange(num_verts_ref), num_corresps)

    logging.info(
        "Computing transformations from %d corresponding verts...", num_corresps
    )
    num_instances = int(len(groupmesh.vertices) // len(referencemesh.vertices))
    verts_instances = np.split(groupmesh.vertices, num_instances)

    transformations = []
    for i, verts_inst in enumerate(verts_instances):
        T_I2R = recover_6D_transform(
            referencemesh.vertices[corresponding_idxs], verts_inst[corresponding_idxs]
        )
        transformations.append(T_I2R)
        logging.info("Instance %d:\n%r", i, T_I2R)

        if args.verify:
            verts_orig = np.ones((num_corresps, 4))
            verts_orig[:, :3] = referencemesh.vertices[corresponding_idxs]
            verts_trafod = (T_I2R @ verts_orig.T).T
            verts_trafod = verts_trafod[:, :3]
            assert np.allclose(
                verts_trafod, verts_inst[corresponding_idxs]
            ), "Verification failed, transformed verts don't match :("

    if args.outcsv:
        logging.info("Writing trafos to %s", args.outcsv)

        outstr = ""
        for trafo in transformations:
            outstr += ",".join([f"{num:.8f}" for num in trafo.flatten()])
            outstr += "\n"

        with open(args.outcsv, "wt") as fh:
            fh.write(outstr)

    logging.info("Done")


def meshes_have_same_metric(groupmesh, referencemesh):
    assert (
        len(groupmesh.vertices) % len(referencemesh.vertices) == 0
    ), "groupmesh doesn't have multiple of referencemesh's vertices ({} vs {}). Make sure the groupmesh contains nothing else!".format(
        len(groupmesh.vertices), len(referencemesh.vertices)
    )

    num_verts = len(referencemesh.vertices)
    verts_ref = referencemesh.vertices

    for offset in range(0, len(groupmesh.vertices) // num_verts):
        verts_inst = groupmesh.vertices[
            offset * num_verts : offset * num_verts + num_verts
        ]
        if not verts_have_same_distances(verts_ref, verts_inst):
            return False
    return True


def verts_have_same_distances(vertsA, vertsB):
    assert len(vertsA) == len(vertsB), "Trying to compare different number of verts!"

    num_verts_to_check = int(min(len(vertsA) * 0.1, 50))
    vert_idxs_to_check = np.random.choice(
        np.arange(num_verts_to_check), replace=False, size=4
    )

    vertsA = vertsA[vert_idxs_to_check]
    vertsB = vertsB[vert_idxs_to_check]

    intra_distsA = []
    intra_distsB = []

    for vert1, vert2 in zip(vertsA, vertsA[::-1]):
        intra_distsA.append(np.linalg.norm(vert1 - vert2))
    for vert1, vert2 in zip(vertsB, vertsB[::-1]):
        intra_distsB.append(np.linalg.norm(vert1 - vert2))

    logging.debug("vertsA distances: %r", intra_distsA)
    logging.debug("vertsB distances: %r", intra_distsB)

    return np.allclose(intra_distsA, intra_distsB, rtol=0.001)


def recover_6D_transform(pts_a, pts_b):
    """
    Returns least squares transform between the two point
    sets pts_a and pts_b: T_b2a (Arun et. al 1987)
    """

    assert np.shape(pts_a) == np.shape(pts_b), "Input data must have same shape"
    assert np.shape(pts_a)[1] == 3, "Expecting points as (N,3) matrix"

    p_centroid = np.mean(pts_a, axis=0).reshape(-1, 1)
    pp_centroid = np.mean(pts_b, axis=0).reshape(-1, 1)

    qs = [np.reshape(pi, (-1, 1)) - p_centroid for pi in pts_a]
    qps = [np.reshape(ppi, (-1, 1)) - pp_centroid for ppi in pts_b]

    H = np.zeros((3, 3))
    for qi, qpi in zip(qs, qps):
        H += qi @ qpi.T

    U, _, Vt = np.linalg.svd(H)
    X = Vt.T @ U.T
    assert np.isclose(np.linalg.det(X), 1), f"Determinant is off!: {np.linalg.det(X)}"

    t = pp_centroid - X @ p_centroid

    T_b2a = np.zeros((4, 4))
    T_b2a[:3, :3] = X
    T_b2a[:3, 3] = t.flatten()

    return T_b2a


if __name__ == "__main__":
    main()
