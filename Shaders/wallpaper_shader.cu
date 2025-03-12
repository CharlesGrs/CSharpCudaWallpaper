// Structure to match C# IconInfo
struct IconInfo
{
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
    int iconCount)
{
    // Calculate pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    // Normalized coordinates in [0,1] range
    float u = (float)x / width;
    float v = (float)y / height;
    
    // Default color - a pleasant gradient background
    float3 color = make_float3(
        0.5f + 0.5f * sinf(time * 0.2f),
        0.5f + 0.5f * sinf(time * 0.1f + 2.0f),
        0.7f
    );
    
    // Distance to mouse position
    float mouseDistance = sqrtf((u - mouseX) * (u - mouseX) + (v - mouseY) * (v - mouseY));
    
    // Create a subtle glow around the mouse cursor
    float mouseFactor = 0.1f / (0.01f + mouseDistance * 0.5f);
    color = make_float3(
        color.x + mouseFactor * 0.5f,
        color.y + mouseFactor * 0.3f,
        color.z + mouseFactor * 0.7f
    );
    
    // Write the final color
    output[y * width + x] = make_float4(color.x, color.y, color.z, 1.0f);
}