# brocken-spectre

Two tiny GPU raytracers rendering the same scene (one red sphere on a sky gradient):

- `raytracer.cu` — CUDA compute kernel, writes a PPM image to disk.
- `raytracer_gl.cpp` — OpenGL fragment shader, renders interactively in a window.

Build instructions below assume [MSYS2](https://www.msys2.org/) on Windows. Install it, then open the **MSYS2 UCRT64** shell.

## raytracer.cu (CUDA)

Requires an NVIDIA GPU and the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed on Windows (CUDA is not available through MSYS2 packages).

`nvcc` on Windows needs MSVC's `cl.exe`, not MinGW. From the **x64 Native Tools Command Prompt for VS**:

```
nvcc raytracer.cu -o raytracer.exe
.\raytracer.exe
```

Produces `out.ppm` (800×600). View it with IrfanView / GIMP, or convert:

```
magick out.ppm out.png
```

## raytracer_gl.cpp (OpenGL)

In the **MSYS2 UCRT64** shell:

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc \
          mingw-w64-ucrt-x86_64-glfw \
          mingw-w64-ucrt-x86_64-glew

g++ raytracer_gl.cpp -o raytracer_gl.exe -lglfw3 -lglew32 -lopengl32
./raytracer_gl.exe
```

Close the window to exit.
