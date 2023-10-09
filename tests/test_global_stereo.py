import os

import oceanmesh as om

# Note: to following file is partof the test data from pyposeidon Tutorial notebooks
# https://github.com/ec-jrc/pyPoseidon/tree/master/Tutorial
# You can get them with
# curl -L url https://www.dropbox.com/sh/nd2b012wrpt6qnq/AAAD7aA_qXztUhlT39YK2yBua?dl=1 > data.zip
fname = os.path.join(os.path.dirname(__file__), "ocean.shp")


def test_global_stereo():
    # it is necessary to define all the coastlines at once:
    # the Shoreline class will the detect the biggest coastline (antartica and define it
    # as the outside boundary)

    EPSG = 4326  # EPSG:4326 or WGS84
    bbox = (-180.00, 180.00, -89.00, 90.00)
    extent = om.Region(extent=bbox, crs=4326)

    min_edge_length = 0.5  # minimum mesh size in domain in meters
    max_edge_length = 2  # maximum mesh size in domain in meters
    shoreline = om.Shoreline(fname, extent.bbox, min_edge_length)
    sdf = om.signed_distance_function(shoreline)
    edge_length0 = om.distance_sizing_function(shoreline, rate=0.11)
    edge_length1 = om.feature_sizing_function(
        shoreline,
        sdf,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        crs=EPSG,
    )

    edge_length = om.compute_minimum([edge_length0, edge_length1])
    edge_length = om.enforce_mesh_gradation(edge_length, gradation=0.09, stereo=True)

    shoreline_stereo = om.Shoreline(fname, extent.bbox, min_edge_length, stereo=True)
    domain = om.signed_distance_function(shoreline_stereo)

    points, cells = om.generate_mesh(domain, edge_length, stereo=True, max_iter=100)

    # remove degenerate mesh faces and other common problems in the mesh
    points, cells = om.make_mesh_boundaries_traversable(points, cells)
    points, cells = om.delete_faces_connected_to_one_face(points, cells)

    # apply a Laplacian smoother
    points, cells = om.laplacian2(points, cells, max_iter=100)

    # plot
    fig, ax, _ = edge_length.plot(
        holding=True,
        plot_colorbar=True,
        stereo=True,
        vmax=max_edge_length,
    )

    ax.triplot(points[:, 0], points[:, 1], cells, color="gray", linewidth=0.5)
    shoreline_stereo.plot(ax=ax)
