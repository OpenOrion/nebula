import zmq
import pickle
from http.client import REQUEST_TIMEOUT
import jax.numpy as jnp
from typing import Literal, Optional, Union
from dataclasses import dataclass, asdict
from nebula.helpers.vector import VectorHelper
from nebula.render.tesselation import Mesh, Tesselator
from nebula.topology.solids import Solids


@dataclass
class BoundingBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


@dataclass
class PartShape:
    vertices: list[list[float]]
    triangles: list[int]
    normals: list[list[float]]
    edges: list[list[list[float]]]


@dataclass
class Part:
    name: str
    id: str
    shape: PartShape
    type: str = "shapes"
    color: str = "#e8b024"
    renderback: bool = True


@dataclass
class Shapes:
    bb: Optional[BoundingBox]
    parts: list[Part]
    name: str = "Group"
    id: str = "/Group"
    loc: Optional[tuple[list[float], tuple[float, float, float, float]]] = None


def get_part(mesh: Mesh, index: int = 0):
    return Part(
        name=f"Part_{index}",
        id=f"/Group/Part_{index}",
        shape=PartShape(
            vertices=mesh.vertices.tolist(),
            triangles=mesh.simplices.flatten().tolist(),
            normals=mesh.normals.tolist(),
            edges=mesh.edges.tolist(),
        ),
    )


def get_bounding_box(vertices: jnp.ndarray):
    return BoundingBox(
        xmin=float(vertices[:, 0].min()),
        xmax=float(vertices[:, 0].max()),
        ymin=float(vertices[:, 1].min()),
        ymax=float(vertices[:, 1].max()),
        zmin=float(vertices[:, 2].min()) or -1e-07,
        zmax=float(vertices[:, 2].max()) or 1e-07,
    )


DEFAULT_CONFIG = {
    "viewer": "",
    "anchor": "right",
    "theme": "light",
    "pinning": False,
    "angular_tolerance": 0.2,
    "deviation": 0.1,
    "edge_accuracy": None,
    "default_color": [232, 176, 36],
    "default_edge_color": "#707070",
    "optimal_bb": False,
    "render_normals": False,
    "render_edges": True,
    "render_mates": False,
    "parallel": False,
    "mate_scale": 1,
    "control": "trackball",
    "up": "Z",
    "axes": False,
    "axes0": False,
    "grid": [False, False, False],
    "ticks": 10,
    "ortho": True,
    "transparent": False,
    "black_edges": False,
    "ambient_intensity": 0.75,
    "direct_intensity": 0.15,
    "reset_camera": True,
    "show_parent": True,
    "show_bbox": False,
    "quaternion": None,
    "target": None,
    "zoom_speed": 1.0,
    "pan_speed": 1.0,
    "rotate_speed": 1.0,
    "collapse": 1,
    "tools": True,
    "timeit": False,
    "js_debug": False,
    "normal_len": 0,
}
ZMQ_PORT = 5555


def connect(context):
    endpoint = f"tcp://localhost:{ZMQ_PORT}"
    socket = context.socket(zmq.REQ)
    socket.connect(endpoint)
    return socket


def send(data):
    context = zmq.Context()
    socket = connect(context)

    msg = pickle.dumps(data, 4)
    print(" sending ... ", end="")
    socket.send(msg)

    retries_left = 3
    while True:
        if (socket.poll(REQUEST_TIMEOUT) & zmq.POLLIN) != 0:
            reply = socket.recv_json()

            if reply["result"] == "success":
                print("done")
            else:
                print("\n", reply["msg"])
            break

        retries_left -= 1

        # Socket is confused. Close and remove it.
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        if retries_left == 0:
            break

        print("Reconnecting to server…")
        # Create new connection
        socket = connect(context)

        print("Resending ...")
        socket.send(msg)


def show(
    item: Union[Solids, Mesh],
    type: Literal["cad", "plot"] = "cad",
    intensity: Optional[jnp.ndarray] = None,
    name: Optional[str] = None,
    file_name: Optional[str] = None,
    **kwargs,
):
    mesh = Tesselator.get_mesh(item) if isinstance(item, Solids) else item

    if type == "cad":
        parts: list[Part] = [get_part(mesh)]

        shapes = Shapes(
            bb=get_bounding_box(mesh.vertices),
            parts=parts,
        )

        states = {"/Group/Part_0": [1, 1]}
        data = {
            "data": dict(shapes=asdict(shapes), states=states),
            "type": "data",
            "config": DEFAULT_CONFIG,
            "count": len(parts),
        }
        send(data)

    else:
        import plotly.graph_objects as go

        Xe = []
        Ye = []
        Ze = []
        for T in mesh.vertices[mesh.simplices]:
            Xe.extend([T[k % 3][0] for k in range(4)] + [None])
            Ye.extend([T[k % 3][1] for k in range(4)] + [None])
            Ze.extend([T[k % 3][2] for k in range(4)] + [None])

        # Create a mesh object
        if intensity is not None:
            intensity = VectorHelper.normalize(intensity)
        plot_mesh = go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=[face[0] for face in mesh.simplices],
            j=[face[1] for face in mesh.simplices],
            k=[face[2] for face in mesh.simplices],
            intensity=intensity,
        )

        lines = go.Scatter3d(
            x=Xe,
            y=Ye,
            z=Ze,
            mode="lines",
            line=dict(color="rgb(70,70,70)", width=1),
        )

        # Create a figure and add the mesh to it
        fig = go.Figure(data=[plot_mesh, lines])

        # Update layout to remove axes and background and make the graph larger
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgba(0,0,0,0)",
            ),
            width=800,  # Set the width of the graph to 800 pixels
            height=600,  # Set the height of the graph to 600 pixels
            title=name,
        )
        fig.update_layout(scene=dict(bgcolor="rgba(0,0,0,0)"))
        if file_name is not None:
            fig.write_image(file_name, scale=6, format="png", engine="kaleido")
        # Display the figure
        fig.show()
