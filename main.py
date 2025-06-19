import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import time
from pyglm import glm

from kingfisher.core.camera import OrbitCamera
from kingfisher.core.loader import load_model

import dearpygui.dearpygui as dpg
import threading

light_intesity = 1.0

# Shader sources
VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

out vec3 FragNormal;
out vec2 FragTexCoord;

uniform mat4 mvp;

void main() {
    FragNormal = normal;
    FragTexCoord = texcoord;
    gl_Position = mvp * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 FragNormal;
in vec2 FragTexCoord;

out vec4 FragColor;

uniform float lightIntensity;
uniform vec3 lightDirection;

void main() {
    vec3 norm = normalize(FragNormal);
    vec3 lightDir = normalize(lightDirection);

    float diffuse = max(dot(norm, -lightDir), 0.0);
    diffuse *= lightIntensity;

    FragColor = vec4(diffuse, diffuse * 0.5, diffuse * 0.2, 1.0);
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_shader_program():
    vertex = compile_shader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
    fragment = compile_shader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    return program



def main():
    if not glfw.init():
        return
    
    def launch_ui():
        def update_light(sender, app_data):
            global light_intesity
            light_intesity = app_data

        dpg.create_context()
        dpg.create_viewport(title="Kingfisher Control Panel", width=400, height=300)
        dpg.setup_dearpygui()

        with dpg.window(label="Controls"):
            dpg.add_text("Settings")
            dpg.add_slider_float(label="Light Intensity", min_value=0.0, max_value=1.0, default_value=1.0, callback=update_light)
            dpg.add_button(label="Reload Model")

        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    threading.Thread(target=launch_ui, daemon=True).start()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(1000, 800, "3D Viewer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    camera = OrbitCamera()
    def scroll_callback(window, xoffset, yoffset):
        camera.zoom(yoffset)
    
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    # Load model
    model_data = load_model("./kingfisher/assets/models/skull.obj")
    vertices = model_data["vertices"]
    faces = model_data["faces"]
    normals = model_data["normals"]
    uvs = model_data["uvs"]

    # Combine vertex attributes: [ position | normal | texcoord ]
    vertex_data = []

    for i in range(len(vertices)):
        pos = vertices[i]
        norm = normals[i] if normals is not None else [0.0, 0.0, 0.0]
        uv = uvs[i] if uvs is not None else [0.0, 0.0]
        vertex_data.extend(pos)
        vertex_data.extend(norm)
        vertex_data.extend(uv)

    vertex_data = np.array(vertex_data, dtype=np.float32)

    # VAO, VBO, EBO setup
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    
    stride = 8 * 4

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)

    glBindVertexArray(0)

    shader_program = create_shader_program()
    glUseProgram(shader_program)
    mvp_loc = glGetUniformLocation(shader_program, "mvp")
    light_loc = glGetUniformLocation(shader_program, "lightIntensity")
    light_dir_loc = glGetUniformLocation(shader_program, "lightDirection")

    last_time = time.time()

    while not glfw.window_should_close(window):
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        glfw.poll_events()

        glfw_x, glfw_y = glfw.get_window_pos(window)
        dpg.set_viewport_pos((glfw_x + 1000, glfw_y))  # Adjust offset as needed


        # Mouse control handling:
        if glfw.get_window_attrib(window, glfw.FOCUSED):
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                xpos, ypos = glfw.get_cursor_pos(window)
                camera.update_mouse(xpos, ypos)
            

        # Close Window (ESC)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        model = glm.mat4(1.0)
        view = camera.get_view_matrix()
        projection = glm.perspective(glm.radians(45.0), 800/600, 0.1, 100.0)
        mvp = projection * view * model

        angle = current_time * 0.5
        light_direction = glm.vec3(glm.cos(angle), -1.0, glm.sin(angle))
        light_direction = glm.normalize(light_direction)

        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp))
        glUniform1f(light_loc, light_intesity)
        glUniform3f(light_dir_loc, light_direction.x, light_direction.y, light_direction.z)


        glUseProgram(shader_program)
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(faces), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glfw.swap_buffers(window)

    dpg.destroy_context()

if __name__ == "__main__":
    main()