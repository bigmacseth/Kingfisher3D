from pyglm import glm

class OrbitCamera:
    def __init__(self, target=None, distance=5.0):
        self.target = target if target else glm.vec3(0.0, 0.0, 0.0)
        self.distance = distance
        self.yaw = -90.0
        self.pitch = 0.0
        self.last_x = 400
        self.last_y = 300
        self.first_mouse = True

        self.position = glm.vec3(0.0, 0.0, distance)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)

    def update_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        sensitivity = 0.3
        xoffset *= sensitivity
        yoffset *= sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        self.pitch = max(-89, min(89.0, self.pitch))

        rad_yaw = glm.radians(self.yaw)
        rad_pitch = glm.radians(self.pitch)

        self.position.x = self.target.x + self.distance * glm.cos(rad_pitch) * glm.cos(rad_yaw)
        self.position.y = self.target.y + self.distance * glm.sin(rad_pitch)
        self.position.z = self.target.z + self.distance * glm.cos(rad_pitch) * glm.sin(rad_yaw)

        self.front = glm.normalize(self.target - self.position)
        self.right = glm.normalize(glm.cross(self.front, self.up))

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)
    
    def zoom(self, yoffset):
        self.distance -= yoffset * 0.5
        self.distance = max(1.0, min(self.distance, 50.0)) # zoom clamp

        # recalculate position immediately after zooming
        rad_yaw = glm.radians(self.yaw)
        rad_pitch = glm.radians(self.pitch)

        self.position.x = self.target.x + self.distance * glm.cos(rad_pitch) * glm.cos(rad_yaw)
        self.position.y = self.target.y + self.distance * glm.sin(rad_pitch)
        self.position.z = self.target.z + self.distance * glm.cos(rad_pitch) * glm.sin(rad_yaw)

        self.front = glm.normalize(self.target - self.position)

    def update_pan(self, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = ypos - self.last_y
        self.last_x = xpos
        self.last_y = ypos

        pan_speed = 0.0005 * self.distance
        self.right = glm.normalize(glm.cross(self.front, self.up))

        self.target += -self.right * xoffset * pan_speed
        self.target += self.up * yoffset * pan_speed
        self.update_position()    
    
    def update_position(self):
        rad_yaw = glm.radians(self.yaw)
        rad_pitch = glm.radians(self.pitch)

        self.position.x = self.target.x + self.distance * glm.cos(rad_pitch) * glm.cos(rad_yaw)
        self.position.y = self.target.y + self.distance * glm.sin(rad_pitch)
        self.position.z = self.target.z + self.distance * glm.cos(rad_pitch) * glm.sin(rad_yaw)

        self.front = glm.normalize(self.target - self.position)
        self.right = glm.normalize(glm.cross(self.front, self.up))
