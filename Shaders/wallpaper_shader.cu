// wallpaper_shader.cu
#include <cuda_runtime.h>

struct IconInfo {
    float X;
    float Y;
    float Width;
    float Height;
    int Selected;
};

extern "C" __global__ void wallpaperShader(
    float4* output,
    int width,
    int height,
    float time,
    float mouseX,
    float mouseY,
    IconInfo* icons,
    int numIcons)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x >= width || y >= height) return;

    // Example shader logic (replace with your actual shader code)
    float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    
    // Example: Simple time-based animation
    color.x = sinf(time + mouseX);
    color.y = cosf(time + mouseY);
    color.z = sinf(time + mouseX + mouseY);

    // Example: Check if pixel is inside an icon
    for (int i = 0; i < numIcons; i++) {
        IconInfo icon = icons[i];
        if (x >= icon.X && x < icon.X + icon.Width &&
            y >= icon.Y && y < icon.Y + icon.Height) {
            // Highlight selected icons
            if (icon.Selected) {
                color = make_float4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow for selected
            } else {
                color = make_float4(0.5f, 0.5f, 0.5f, 1.0f); // Gray for icons
            }
            }
    }

    output[idx] = color;
}