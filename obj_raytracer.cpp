// OpenGL raytracer that loads an OBJ mesh and renders it via brute-force
// ray-triangle intersection in the fragment shader. Triangles live in an SSBO.
//
// Build (MSYS2 UCRT64):
//   g++ raytracer_obj.cpp -o raytracer_obj.exe -lglfw3 -lglew32 -lopengl32
// Run:
//   ./raytracer_obj.exe path/to/model.obj
//
// Linux: g++ raytracer_obj.cpp -o raytracer_obj -lglfw -lGLEW -lGL

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

// Parse 'v x y z' and 'f a b c ...' lines. Fan-triangulates polygons.
// Returns a flat array: 9 floats per triangle (3 verts × xyz).
// Also fills bmin/bmax with the bounding box.
static std::vector<float> load_obj(const char* path, float bmin[3], float bmax[3]) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Can't open %s\n", path); exit(1); }

    std::vector<float> verts;  // flat xyz
    std::vector<float> tris;   // flat 9-floats-per-triangle

    for (int i = 0; i < 3; i++) { bmin[i] = 1e30f; bmax[i] = -1e30f; }

    std::string line;
    while (std::getline(f, line)) {
        if (line.size() < 2) continue;

        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (sscanf(line.c_str() + 2, "%f %f %f", &x, &y, &z) != 3) continue;
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
            if (x < bmin[0]) bmin[0] = x;  if (x > bmax[0]) bmax[0] = x;
            if (y < bmin[1]) bmin[1] = y;  if (y > bmax[1]) bmax[1] = y;
            if (z < bmin[2]) bmin[2] = z;  if (z > bmax[2]) bmax[2] = z;
        }
        else if (line[0] == 'f' && line[1] == ' ') {
            // Collect vertex indices from this face (first number before any '/').
            std::vector<int> idx;
            std::istringstream iss(line.substr(2));
            std::string tok;
            while (iss >> tok) {
                int v = atoi(tok.c_str());                  // OBJ is 1-indexed
                if (v < 0) v = (int)(verts.size() / 3) + v + 1;  // negative = relative
                idx.push_back(v - 1);
            }
            // Fan-triangulate (works for tris and convex quads/n-gons).
            for (size_t i = 1; i + 1 < idx.size(); i++) {
                int a = idx[0], b = idx[i], c = idx[i + 1];
                for (int vi : {a, b, c}) {
                    tris.push_back(verts[vi * 3 + 0]);
                    tris.push_back(verts[vi * 3 + 1]);
                    tris.push_back(verts[vi * 3 + 2]);
                }
            }
        }
    }

    printf("Loaded %zu vertices, %zu triangles\n", verts.size() / 3, tris.size() / 9);
    return tris;
}

const char* VS = R"GLSL(
#version 430 core
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

const char* FS = R"GLSL(
#version 430 core
out vec4 frag_color;

layout(std430, binding = 0) readonly buffer Triangles { float data[]; };

uniform int   triangle_count;
uniform vec2  resolution;
uniform vec3  cam_pos;
uniform vec3  cam_target;

// Möller-Trumbore. Returns hit distance, writes outward normal.
float hit_tri(vec3 ro, vec3 rd, vec3 a, vec3 b, vec3 c, out vec3 n) {
    vec3 e1 = b - a;
    vec3 e2 = c - a;
    vec3 p = cross(rd, e2);
    float det = dot(e1, p);
    if (abs(det) < 1e-8) return -1.0;
    float inv = 1.0 / det;
    vec3 t = ro - a;
    float u = dot(t, p) * inv;
    if (u < 0.0 || u > 1.0) return -1.0;
    vec3 q = cross(t, e1);
    float v = dot(rd, q) * inv;
    if (v < 0.0 || u + v > 1.0) return -1.0;
    float dist = dot(e2, q) * inv;
    if (dist < 1e-4) return -1.0;
    n = normalize(cross(e1, e2));
    return dist;
}

void main() {
    vec2 ndc = (gl_FragCoord.xy / resolution) * 2.0 - 1.0;
    float aspect = resolution.x / resolution.y;

    // Camera basis (look-at).
    vec3 fwd   = normalize(cam_target - cam_pos);
    vec3 right = normalize(cross(fwd, vec3(0, 1, 0)));
    vec3 up    = cross(right, fwd);

    vec3 ro = cam_pos;
    vec3 rd = normalize(fwd * 1.5 + right * ndc.x * aspect + up * ndc.y);

    float best_t = 1e30;
    vec3  best_n = vec3(0);
    bool  hit    = false;

    for (int i = 0; i < triangle_count; i++) {
        int o = i * 9;
        vec3 a = vec3(data[o + 0], data[o + 1], data[o + 2]);
        vec3 b = vec3(data[o + 3], data[o + 4], data[o + 5]);
        vec3 c = vec3(data[o + 6], data[o + 7], data[o + 8]);
        vec3 n;
        float t = hit_tri(ro, rd, a, b, c, n);
        if (t > 0.0 && t < best_t) { best_t = t; best_n = n; hit = true; }
    }

    vec3 color;
    if (hit) {
        vec3 light = normalize(vec3(1, 1, 0.5));
        // Two-sided shading in case the model's winding is inconsistent.
        float diff = max(abs(dot(best_n, light)), 0.0);
        color = vec3(0.8, 0.6, 0.4) * (diff * 0.8 + 0.2);
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
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "shader error: %s\n", log);
        exit(1);
    }
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.obj\n", argv[0]); return 1; }

    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(800, 600, "OBJ Raytracer", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return 1;

    // Load mesh and figure out where to put the camera.
    float bmin[3], bmax[3];
    auto tris = load_obj(argv[1], bmin, bmax);
    int tri_count = (int)(tris.size() / 9);

    float cx = (bmin[0] + bmax[0]) * 0.5f;
    float cy = (bmin[1] + bmax[1]) * 0.5f;
    float cz = (bmin[2] + bmax[2]) * 0.5f;
    float dx = bmax[0] - bmin[0], dy = bmax[1] - bmin[1], dz = bmax[2] - bmin[2];
    float diag = sqrtf(dx*dx + dy*dy + dz*dz);
    float cam_dist = diag * 1.5f;

    // Upload triangles to GPU as an SSBO.
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 tris.size() * sizeof(float), tris.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    GLuint vs = compile(GL_VERTEX_SHADER, VS);
    GLuint fs = compile(GL_FRAGMENT_SHADER, FS);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    glUseProgram(prog);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLint loc_res    = glGetUniformLocation(prog, "resolution");
    GLint loc_count  = glGetUniformLocation(prog, "triangle_count");
    GLint loc_cpos   = glGetUniformLocation(prog, "cam_pos");
    GLint loc_ctgt   = glGetUniformLocation(prog, "cam_target");

    glUniform1i(loc_count, tri_count);
    glUniform3f(loc_ctgt, cx, cy, cz);

    while (!glfwWindowShouldClose(win)) {
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glUniform2f(loc_res, (float)w, (float)h);

        // Orbit camera once per ~6 seconds so you can see the whole model.
        float t = (float)glfwGetTime() * 1.0f;
        glUniform3f(loc_cpos,
                    cx + cam_dist * sinf(t),
                    cy + cam_dist * 0.3f,
                    cz + cam_dist * cosf(t));

        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}