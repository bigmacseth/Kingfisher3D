import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import time
from pyglm import glm

from kingfisher.core.camera import OrbitCamera
from kingfisher.core.loader import load_model
from kingfisher.core.texture import load_texture

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
uniform mat4 model;
uniform mat3 normalMatrix;

void main() {
    FragNormal = normalMatrix * normal;
    FragTexCoord = texcoord;
    gl_Position = mvp * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 FragNormal;
in vec2 FragTexCoord;

out vec4 FragColor;

uniform sampler2D texture1;
uniform float lightIntensity;
uniform vec3 lightDirection;

void main() {
    vec3 norm = normalize(FragNormal);
    vec3 lightDir = normalize(lightDirection);
    vec4 texColor = texture(texture1, FragTexCoord);

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

pending_model_path = None 

def main():
    if not glfw.init():
        return
    
    global pending_model_path

    def launch_ui():
        def update_light(sender, app_data):
            global light_intesity
            light_intesity = app_data

        def load_model_callback(sender, app_data):
            global pending_model_path
            pending_model_path = app_data['file_path_name']

        dpg.create_context()
        dpg.create_viewport(title="Kingfisher Control Panel", width=400, height=300)
        dpg.setup_dearpygui()

        with dpg.window(label="Controls"):
            dpg.add_text("Settings")
            dpg.add_slider_float(label="Light Intensity", min_value=0.0, max_value=1.0, default_value=1.0, callback=update_light)
            dpg.add_button(label='Load Model', callback=lambda: dpg.show_item('model_file_picker'))

            with dpg.file_dialog(directory_selector=False, show=False, callback=load_model_callback, tag='model_file_picker', width=400, height=300):
                dpg.add_file_extension(".obj", color=(150, 255, 150, 255))
                dpg.add_file_extension(".*")

            dpg.add_slider_float(label="Rotate X", tag="rotate_x", min_value=-180, max_value=180, default_value=0)
            dpg.add_slider_float(label="Rotate Y", tag="rotate_y", min_value=-180, max_value=180, default_value=0)
            dpg.add_slider_float(label="Rotate Z", tag="rotate_z", min_value=-180, max_value=180, default_value=0)


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
    model_data = load_model("./kingfisher/assets/models/fish.obj")
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

    model_loc = glGetUniformLocation(shader_program, "model")
    normal_matrix_loc = glGetUniformLocation(shader_program, "normalMatrix")


    last_time = time.time()
    is_middle_drag = False

    while not glfw.window_should_close(window):
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        glfw.poll_events()

        
        if pending_model_path is not None:
            try:
                # Unbind VAO/VBOs
                glBindVertexArray(0)

                # Reload Model
                new_model_data = load_model(pending_model_path)
                new_vertices = new_model_data["vertices"]
                new_faces = new_model_data["faces"]
                new_normals = new_model_data["normals"]
                new_uvs = new_model_data["uvs"]

                # Prepare new vertex data
                new_vertex_data = []
                for i in range(len(new_vertices)):
                    pos = new_vertices[i]
                    norm = new_normals[i] if new_normals is not None else [0.0, 0.0, 0.0]
                    uv = new_uvs[i] if new_uvs is not None else [0.0, 0.0]
                    new_vertex_data.extend(pos)
                    new_vertex_data.extend(norm)
                    new_vertex_data.extend(uv)

                new_vertex_data = np.array(new_vertex_data, dtype=np.float32)

                # Update GPU buffers
                glBindBuffer(GL_ARRAY_BUFFER, VBO)
                glBufferData(GL_ARRAY_BUFFER, new_vertex_data.nbytes, new_vertex_data, GL_STATIC_DRAW)

                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, new_faces.nbytes, new_faces, GL_STATIC_DRAW)

                # Update face count for drawing
                faces = new_faces

            except Exception as e:
                print(f"Failed to load model: {e}")

            # Clear the flag
            pending_model_path = None


        glfw_x, glfw_y = glfw.get_window_pos(window)
        dpg.set_viewport_pos((glfw_x + 1000, glfw_y))  # Adjust offset as needed


        # Mouse control handling:
        if glfw.get_window_attrib(window, glfw.FOCUSED):
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                if not is_middle_drag:
                    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
                    is_middle_drag = True

                xpos, ypos = glfw.get_cursor_pos(window)
                camera.update_mouse(xpos, ypos)

            elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
                if not is_middle_drag:
                    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
                    is_middle_drag = True

                xpos, ypos = glfw.get_cursor_pos(window)
                camera.update_pan(xpos, ypos)

            elif is_middle_drag:
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
                camera.first_mouse = True
                is_middle_drag = False
            

        # Close Window (ESC)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rotate_x = glm.radians(dpg.get_value("rotate_x"))
        rotate_y = glm.radians(dpg.get_value("rotate_y"))
        rotate_z = glm.radians(dpg.get_value("rotate_z"))

        model = glm.mat4(1.0)
        model = glm.rotate(model, rotate_x, glm.vec3(1, 0, 0))
        model = glm.rotate(model, rotate_y, glm.vec3(0, 1, 0))
        model = glm.rotate(model, rotate_z, glm.vec3(0, 0, 1))

        normal_matrix = glm.mat3(glm.transpose(glm.inverse(model)))

        


        view = camera.get_view_matrix()
        projection = glm.perspective(glm.radians(45.0), 800/600, 0.1, 100.0)
        mvp = projection * view * model

        angle = current_time * 0.5
        light_direction = glm.vec3(glm.cos(angle), -1.0, glm.sin(angle))
        light_direction = glm.normalize(light_direction)

        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp))
        glUniform1f(light_loc, light_intesity)
        glUniform3f(light_dir_loc, light_direction.x, light_direction.y, light_direction.z)
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp))
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix3fv(normal_matrix_loc, 1, GL_FALSE, glm.value_ptr(normal_matrix))



        glUseProgram(shader_program)
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(faces), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glfw.swap_buffers(window)

    dpg.destroy_context()

if __name__ == "__main__":
    main()