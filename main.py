import ctypes
import math
from dataclasses import dataclass, field
from pathlib import Path

import glfw
import glm
import numpy as np
from OpenGL.GL import *

WINDOW_W, WINDOW_H = 800, 600
COMPUTE_W, COMPUTE_H = 200, 150

C = 299_792_458.0
G = 6.67430e-11

GRID_SIZE = 50                    #tamanho do grid
GRID_SPACING = 1e10               #espacamento entre as linhas
GRID_DROP = 3e10                  #joga o grid pra baixo pra camera ficar em cima

TIME_SCALE = 5000.0               # 1 segundo na vida real sao 5000 segundos simulados
PHYSICS_SUBSTEPS = 20             # menos passos a orbita e mais estavel
START_IN_ORBIT = True             # corpos comecam em orbita

SHADER_DIR = Path(__file__).parent 


def schwarzschild_radius(mass):
    """Event horizon radius: r_s = 2GM / c^2"""
    return 2.0 * G * mass / C**2


@dataclass
class Body:
    position: np.ndarray                 
    radius: float
    color: tuple 
    mass: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fixed: bool = False


SAGA_MASS = 8.54e36                      # massa do buraco negro Sargetarius A*
SAGA_RS = schwarzschild_radius(SAGA_MASS)
SUN_MASS = 1.98892e30


def make_bodies():
    bodies = [
        Body(np.array([4e11, 0.0, 0.0]), 4e10, (0.0, 0.0, 1.0, 1.0), SUN_MASS),   # yellow star
        Body(np.array([0.0, 0.0, 4e11]), 4e10, (1, 0, 0, 1), SUN_MASS),   # red star
        Body(np.zeros(3), SAGA_RS, (0, 0, 0, 1), SAGA_MASS, fixed=True),  # black hole
    ]
    if START_IN_ORBIT:
        # orbita circular
        v = math.sqrt(G * SAGA_MASS / 4e11)
        bodies[0].velocity = np.array([0.0, 0.0, v])
        bodies[1].velocity = np.array([-v, 0.0, 0.0])
    return bodies


def step_gravity(bodies, dt):
    # gravidade newtoniana entre os corpos
    h = dt / PHYSICS_SUBSTEPS
    for _ in range(PHYSICS_SUBSTEPS):
        for body in bodies:
            if body.fixed:
                continue
            accel = np.zeros(3)
            for other in bodies:
                if other is body:
                    continue
                offset = other.position - body.position
                dist = np.linalg.norm(offset)
                accel += G * other.mass * offset / dist**3 
            body.velocity += accel * h
        for body in bodies:
            if not body.fixed:
                body.position += body.velocity * h


# camera orbita ao redor da origem
class Camera:
    def __init__(self):
        self.radius = 6.34194e10
        self.min_radius = 1e10
        self.max_radius = 1e12
        self.azimuth = 0.0
        self.elevation = math.pi / 2
        self.orbit_speed = 0.04
        self.zoom_speed = 25e9
        self.dragging = False
        self.last_x = 0.0
        self.last_y = 0.0

    def position(self):
        e = min(max(self.elevation, 0.01), math.pi - 0.01)
        return glm.vec3(
            self.radius * math.sin(e) * math.cos(self.azimuth),
            self.radius * math.cos(e),
            self.radius * math.sin(e) * math.sin(self.azimuth),
        )

    def on_mouse_move(self, x, y):
        if self.dragging:
            self.azimuth += (x - self.last_x) * self.orbit_speed
            self.elevation -= (y - self.last_y) * self.orbit_speed
            self.elevation = min(max(self.elevation, 0.01), math.pi - 0.01)
        self.last_x, self.last_y = x, y

    def on_scroll(self, yoffset):
        self.radius -= yoffset * self.zoom_speed
        self.radius = min(max(self.radius, self.min_radius), self.max_radius)

def compile_shader(source, kind):
    shader = glCreateShader(kind)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError("Shader compile error:\n" + glGetShaderInfoLog(shader).decode())
    return shader


def link_program(*shaders):
    program = glCreateProgram()
    for s in shaders:
        glAttachShader(program, s)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError("Program link error:\n" + glGetProgramInfoLog(program).decode())
    for s in shaders:
        glDeleteShader(s)
    return program


def make_ubo(binding, size):
    """Create a uniform buffer and attach it to a binding point used by the shader."""
    ubo = glGenBuffers(1)
    glBindBuffer(GL_UNIFORM_BUFFER, ubo)
    glBufferData(GL_UNIFORM_BUFFER, size, None, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_UNIFORM_BUFFER, binding, ubo)
    return ubo


def upload_ubo(ubo, data):
    glBindBuffer(GL_UNIFORM_BUFFER, ubo)
    glBufferSubData(GL_UNIFORM_BUFFER, 0, data.nbytes, data)


# shaders para o grid e para o quad de tela cheia (o shader esta em geodesic.comp)
GRID_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 viewProj;
void main() {
    gl_Position = viewProj * vec4(aPos, 1.0);
}
"""

GRID_FRAG = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.5, 0.5, 0.5, 0.7);   // translucent grey lines
}
"""

QUAD_VERT = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

QUAD_FRAG = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D screenTexture;
void main() {
    FragColor = texture(screenTexture, TexCoord);
}
"""

# Layouts de memória dos três blocos uniformes em geodesic.comp (regras std140:

# todos vetores estao armazenados em um slot de 16 bytes por isso o padding
CAMERA_DTYPE = np.dtype([
    ("pos", "f4", 3), ("_p0", "f4"),
    ("right", "f4", 3), ("_p1", "f4"),
    ("up", "f4", 3), ("_p2", "f4"),
    ("forward", "f4", 3), ("_p3", "f4"),
    ("tan_half_fov", "f4"), ("aspect", "f4"), ("moving", "i4"), ("_p4", "i4"),
])
OBJECTS_DTYPE = np.dtype([
    ("count", "i4"), ("_pad", "i4", 3),
    ("pos_radius", "f4", (16, 4)),
    ("color", "f4", (16, 4)),
    ("mass", "f4", (16, 4)),
])


def build_grid_indices():
    """Line segments: each cell contributes one line to the right and one forward."""
    n = GRID_SIZE
    i = (np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]).ravel()
    return np.stack([i, i + 1, i, i + n + 1], axis=1).ravel().astype(np.uint32)


def build_grid_vertices(bodies):

    n = GRID_SIZE
    coords = (np.arange(n + 1) - n // 2) * GRID_SPACING
    xs, zs = np.meshgrid(coords, coords)             # shape (rows = z, cols = x)
    ys = np.zeros_like(xs)

    for body in bodies:
        rs = schwarzschild_radius(body.mass)
        dist = np.hypot(xs - body.position[0], zs - body.position[2])
        outside = 2.0 * np.sqrt(rs * np.maximum(dist - rs, 0.0))
        inside = 2.0 * rs                            # inside the horizon: flat pit
        ys += np.where(dist > rs, outside, inside) - GRID_DROP

    return np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)


class Renderer:
    def __init__(self, window):
        self.window = window

        # programs
        self.grid_prog = link_program(
            compile_shader(GRID_VERT, GL_VERTEX_SHADER),
            compile_shader(GRID_FRAG, GL_FRAGMENT_SHADER))
        self.quad_prog = link_program(
            compile_shader(QUAD_VERT, GL_VERTEX_SHADER),
            compile_shader(QUAD_FRAG, GL_FRAGMENT_SHADER))
        comp_source = (SHADER_DIR / "geodesic.comp").read_text()
        self.compute_prog = link_program(compile_shader(comp_source, GL_COMPUTE_SHADER))

        self.camera_ubo = make_ubo(1, CAMERA_DTYPE.itemsize)
        self.disk_ubo = make_ubo(2, 16)
        self.objects_ubo = make_ubo(3, OBJECTS_DTYPE.itemsize)

        self._setup_quad()
        self._setup_grid()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    # setup
    def _setup_quad(self):
        quad = np.array([
            # x, y,   u, v
            -1,  1,   0, 1,
            -1, -1,   0, 0,
             1, -1,   1, 0,
            -1,  1,   0, 1,
             1, -1,   1, 0,
             1,  1,   1, 1,
        ], dtype=np.float32)
        self.quad_vao = glGenVertexArrays(1)
        glBindVertexArray(self.quad_vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

        # a textura que o shader usa
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, COMPUTE_W, COMPUTE_H, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)

    def _setup_grid(self):
        self.grid_vao = glGenVertexArrays(1)
        glBindVertexArray(self.grid_vao)

        self.grid_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.grid_vbo)
        glBufferData(GL_ARRAY_BUFFER, (GRID_SIZE + 1) ** 2 * 12, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        indices = build_grid_indices()
        self.grid_index_count = len(indices)
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # uploads por frame
    def upload_camera(self, camera, moving):
        pos = camera.position()
        forward = glm.normalize(-pos)                        # olhando para a origem
        right = glm.normalize(glm.cross(forward, glm.vec3(0, 1, 0)))
        up = glm.cross(right, forward)

        data = np.zeros(1, dtype=CAMERA_DTYPE)
        data["pos"][0] = tuple(pos)
        data["right"][0] = tuple(right)
        data["up"][0] = tuple(up)
        data["forward"][0] = tuple(forward)
        data["tan_half_fov"] = math.tan(math.radians(60.0 / 2))
        data["aspect"] = WINDOW_W / WINDOW_H
        data["moving"] = int(moving)
        upload_ubo(self.camera_ubo, data)

    def upload_disk(self):
        inner = SAGA_RS * 2.2   # borda interna do disco de acreção
        outer = SAGA_RS * 5.2   # borda externa do disco de acreção
        data = np.array([inner, outer, 2.0, 1e9], dtype=np.float32)
        upload_ubo(self.disk_ubo, data)

    def upload_objects(self, bodies):
        data = np.zeros(1, dtype=OBJECTS_DTYPE)
        n = min(len(bodies), 16)
        data["count"] = n
        for i, b in enumerate(bodies[:n]):
            data["pos_radius"][0, i] = (*b.position, b.radius)
            data["color"][0, i] = b.color
            data["mass"][0, i, 0] = b.mass
        upload_ubo(self.objects_ubo, data)

    def upload_grid(self, bodies):
        verts = build_grid_vertices(bodies)
        glBindBuffer(GL_ARRAY_BUFFER, self.grid_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

    # funcao para o desenho
    def draw_grid(self, camera):
        view = glm.lookAt(camera.position(), glm.vec3(0), glm.vec3(0, 1, 0))
        proj = glm.perspective(glm.radians(60.0), COMPUTE_W / COMPUTE_H, 1e9, 1e14)
        view_proj = np.array((proj * view).to_list(), dtype=np.float32)  # column-major

        glUseProgram(self.grid_prog)
        glUniformMatrix4fv(glGetUniformLocation(self.grid_prog, "viewProj"),
                           1, GL_FALSE, view_proj)
        glBindVertexArray(self.grid_vao)
        glDrawElements(GL_LINES, self.grid_index_count, GL_UNSIGNED_INT, None)

    def run_raytracer(self):
        glUseProgram(self.compute_prog)
        glBindImageTexture(0, self.texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8)
        glDispatchCompute(math.ceil(COMPUTE_W / 16), math.ceil(COMPUTE_H / 16), 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT)

    def draw_quad(self):
        glUseProgram(self.quad_prog)
        glBindVertexArray(self.quad_vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(glGetUniformLocation(self.quad_prog, "screenTexture"), 0)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def render(self, camera, bodies):
        width, height = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT)

        self.upload_grid(bodies)
        self.draw_grid(camera)               # distorcao do grid

        self.upload_camera(camera, camera.dragging)
        self.upload_disk()
        self.upload_objects(bodies)
        self.run_raytracer()                 # gpu traca a luz 
        self.draw_quad()                     # resultados desenhados sob o grid



def main():
    if not glfw.init():
        raise RuntimeError("GLFW failed to initialise")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(WINDOW_W, WINDOW_H, "Black Hole", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL 4.3 window (does your GPU support it?)")
    glfw.make_context_current(window)
    print("OpenGL", glGetString(GL_VERSION).decode())

    camera = Camera()
    bodies = make_bodies()
    gravity = {"on": False}

    # callbacks
    def on_mouse_button(win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            camera.dragging = (action == glfw.PRESS)
            if camera.dragging:
                camera.last_x, camera.last_y = glfw.get_cursor_pos(win)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            gravity["on"] = (action == glfw.PRESS)

    def on_cursor(win, x, y):
        camera.on_mouse_move(x, y)

    def on_scroll(win, xoffset, yoffset):
        camera.on_scroll(yoffset)

    def on_key(win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_G:
            gravity["on"] = not gravity["on"]
            print("[INFO] Gravity", "ON" if gravity["on"] else "OFF")
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)

    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    renderer = Renderer(window)

    last_time = glfw.get_time()
    while not glfw.window_should_close(window):
        now = glfw.get_time()
        dt = now - last_time
        last_time = now

        if gravity["on"]:
            step_gravity(bodies, dt * TIME_SCALE)

        renderer.render(camera, bodies)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()