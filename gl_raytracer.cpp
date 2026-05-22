// Minimal OpenGL raytracer: same scene as the CUDA version, rendered as a
// fullscreen fragment shader. One red sphere on a sky gradient.
//
// Build (Ubuntu/Debian):
//   sudo apt install libglfw3-dev libglew-dev
//   g++ raytracer_gl.cpp -o raytracer_gl -lglfw -lGLEW -lGL

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>

const char* VS = R"GLSL(
#version 330 core
// Fullscreen triangle from gl_VertexID — no VBO needed.
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

const char* FS = R"GLSL(
#version 330 core
out vec4 frag_color;
uniform vec2 resolution;

float hit_sphere(vec3 ro, vec3 rd, vec3 c, float r) {
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float k = dot(oc, oc) - r * r;
    float h = b * b - k;
    if (h < 0.0) return -1.0;
    return -b - sqrt(h);
}

void main() {
    vec2 ndc = (gl_FragCoord.xy / resolution) * 2.0 - 1.0;
    float aspect = resolution.x / resolution.y;

    vec3 ro = vec3(0.0);
    vec3 rd = normalize(vec3(ndc.x * aspect, ndc.y, -1.5));

    vec3 sphere = vec3(0.0, 0.0, -3.0);
    float t = hit_sphere(ro, rd, sphere, 1.0);

    vec3 color;
    if (t > 0.0) {
        vec3 hit = ro + rd * t;
        vec3 n = normalize(hit - sphere);
        vec3 light = normalize(vec3(1.0, 1.0, 0.5));
        float diff = max(dot(n, light), 0.0);
        color = vec3(0.8, 0.3, 0.3) * (diff * 0.8 + 0.2);
    } else {
        float s = 0.5 * (rd.y + 1.0);
        color = mix(vec3(1.0), vec3(0.5, 0.7, 1.0), s);
    }

    frag_color = vec4(color, 1.0);
}
)GLSL";

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "shader error: %s\n", log);
    }
    return s;
}

int main() {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* win = glfwCreateWindow(800, 600, "GPU Raytracer", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return 1;

    GLuint vs = compile(GL_VERTEX_SHADER, VS);
    GLuint fs = compile(GL_FRAGMENT_SHADER, FS);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLint res_loc = glGetUniformLocation(prog, "resolution");
    glUseProgram(prog);

    while (!glfwWindowShouldClose(win)) {
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glUniform2f(res_loc, (float)w, (float)h);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}