#include <cstdio>
#include <cuda_runtime.h>

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float a=0, float b=0, float c=0) : x(a), y(b), z(c) {}
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    __host__ __device__ Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
};

__device__ float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ Vec3 normalize(Vec3 v) { return v * rsqrtf(dot(v, v)); }

// Returns t (distance) of nearest hit, or -1 if miss.
__device__ float hit_sphere(Vec3 ro, Vec3 rd, Vec3 center, float radius) {
    Vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float h = b*b - c;
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

__global__ void render(unsigned char* fb, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // NDC coords with aspect correction
    float aspect = (float)w / (float)h;
    float u = (2.0f * x / w - 1.0f) * aspect;
    float v = 1.0f - 2.0f * y / h;

    Vec3 ro(0.0f, 0.0f, 0.0f);
    Vec3 rd = normalize(Vec3(u, v, -1.5f));

    Vec3 sphere_center(0.0f, 0.0f, -3.0f);
    float t = hit_sphere(ro, rd, sphere_center, 1.0f);

    Vec3 color;
    if (t > 0.0f) {
        Vec3 hit = ro + rd * t;
        Vec3 normal = normalize(hit - sphere_center);
        Vec3 light = normalize(Vec3(1.0f, 1.0f, 0.5f));
        float diff = fmaxf(dot(normal, light), 0.0f);
        color = Vec3(0.8f, 0.3f, 0.3f) * (diff * 0.8f + 0.2f);
    } else {
        // sky gradient
        float s = 0.5f * (rd.y + 1.0f);
        color = Vec3(1.0f, 1.0f, 1.0f) * (1.0f - s) + Vec3(0.5f, 0.7f, 1.0f) * s;
    }

    int i = (y * w + x) * 3;
    fb[i+0] = (unsigned char)(255.0f * fminf(color.x, 1.0f));
    fb[i+1] = (unsigned char)(255.0f * fminf(color.y, 1.0f));
    fb[i+2] = (unsigned char)(255.0f * fminf(color.z, 1.0f));
}

int main() {
    const int W = 800, H = 600;
    size_t size = W * H * 3;

    unsigned char* fb;
    cudaMallocManaged(&fb, size);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    render<<<grid, block>>>(fb, W, H);
    cudaDeviceSynchronize();

    FILE* f = fopen("out.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(fb, 1, size, f);
    fclose(f);

    cudaFree(fb);
    printf("Wrote out.ppm\n");
    return 0;
}
