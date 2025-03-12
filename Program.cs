using System;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Diagnostics;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using Timer = System.Windows.Forms.Timer;

namespace InteractiveWallpaper
{
    public class Program
    {
        [DllImport("user32.dll", SetLastError = true)]
        public static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern IntPtr SetParent(IntPtr hWndChild, IntPtr hWndNewParent);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool GetWindowRect(IntPtr hwnd, out RECT lpRect);

        [StructLayout(LayoutKind.Sequential)]
        public struct RECT
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            // Create and run the wallpaper form
            Application.Run(new WallpaperForm());
        }
    }

    public class WallpaperForm : Form
    {
        private CudaContext cudaContext;
        private CUmodule cudaModule;
        private CudaKernel shaderKernel; // Changed from CUfunction to CudaKernel
        private CudaDeviceVariable<float4> outputBuffer;
        private CudaDeviceVariable<IconInfo> iconInfoBuffer;

        private Timer renderTimer;
        private Stopwatch stopwatch;

        [StructLayout(LayoutKind.Sequential)]
        struct IconInfo
        {
            public float X;
            public float Y;
            public float Width;
            public float Height;
            public int Selected;
        }

        private IconInfo[] desktopIcons;

        public WallpaperForm()
        {
            // Set up form properties to work as wallpaper
            FormBorderStyle = FormBorderStyle.None;
            WindowState = FormWindowState.Maximized;
            ShowInTaskbar = false;

            // Make form transparent to mouse events
            SetStyle(ControlStyles.Selectable, false);

            // Initialize CUDA and load shader
            InitializeCuda();

            // Set up render timer
            renderTimer = new Timer
            {
                Interval = 16 // ~60 FPS
            };
            renderTimer.Tick += RenderFrame;

            stopwatch = new Stopwatch();
            stopwatch.Start();

            // Set form as wallpaper (child of desktop)
            SetAsWallpaper();

            // Start rendering
            renderTimer.Start();

            // Set up mouse tracking
            this.MouseMove += WallpaperForm_MouseMove;
        }

        private void SetAsWallpaper()
        {
            // Find desktop window (Progman)
            IntPtr progman = Program.FindWindow("Progman", null);

            // Set our form as a child of the desktop
            Program.SetParent(this.Handle, progman);

            // Get desktop dimensions and resize form
            Program.GetWindowRect(progman, out Program.RECT rect);
            this.Size = new Size(rect.Right - rect.Left, rect.Bottom - rect.Top);
            this.Location = new Point(rect.Left, rect.Top);
        }

        private void InitializeCuda()
        {
            try
            {
                // Initialize CUDA context using first device
                cudaContext = new CudaContext(0);

                // Load compiled PTX module (shader code)
                string ptxPath = "wallpaper_shader.ptx";
                if (!System.IO.File.Exists(ptxPath))
                {
                    MessageBox.Show($"PTX file not found: {ptxPath}", "Error", 
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                    Application.Exit();
                    return;
                }
                cudaModule = cudaContext.LoadModule(ptxPath);

                // Create the shader kernel (modern approach)
                shaderKernel = new CudaKernel("wallpaperShader", cudaModule, cudaContext);

                // Allocate device memory for output
                outputBuffer =
                    new CudaDeviceVariable<float4>(Screen.PrimaryScreen.Bounds.Width *
                                                   Screen.PrimaryScreen.Bounds.Height);

                // Initialize desktop icon information (mock data for now)
                desktopIcons = new IconInfo[]
                {
                    new IconInfo { X = 20, Y = 20, Width = 64, Height = 64, Selected = 0 },
                    new IconInfo { X = 20, Y = 104, Width = 64, Height = 64, Selected = 0 },
                    new IconInfo { X = 20, Y = 188, Width = 64, Height = 64, Selected = 0 }
                };

                // Allocate and copy icon info to device
                iconInfoBuffer = new CudaDeviceVariable<IconInfo>(desktopIcons.Length);
                iconInfoBuffer.CopyToDevice(desktopIcons);

                // Create a bitmap for rendering CUDA output
                CreateOutputBitmap();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"CUDA initialization error: {ex.Message}", "Error", MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                Application.Exit();
            }
        }

        private Bitmap outputBitmap;
        private IntPtr outputBitmapData;
        private int stride;

        private void CreateOutputBitmap()
        {
            int width = Screen.PrimaryScreen.Bounds.Width;
            int height = Screen.PrimaryScreen.Bounds.Height;

            // Create bitmap for rendering
            outputBitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            // Lock bitmap to get direct pointer to bitmap data
            var bitmapData = outputBitmap.LockBits(
                new Rectangle(0, 0, width, height),
                System.Drawing.Imaging.ImageLockMode.ReadWrite,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            outputBitmapData = bitmapData.Scan0;
            stride = bitmapData.Stride;

            // Don't unlock yet - we'll keep this locked for performance
        }

        private float mouseX = 0, mouseY = 0;

        private void WallpaperForm_MouseMove(object sender, MouseEventArgs e)
        {
            // Update mouse position for shader
            mouseX = (float)e.X / Width;
            mouseY = (float)e.Y / Height;
        }

        private void RenderFrame(object sender, EventArgs e)
        {
            // Get time for animation
            float time = (float)stopwatch.Elapsed.TotalSeconds;

            try
            {
                // Set up kernel parameters
                int width = Screen.PrimaryScreen.Bounds.Width;
                int height = Screen.PrimaryScreen.Bounds.Height;

                // Configure kernel launch dimensions
                dim3 blockSize = new dim3(16, 16, 1);
                dim3 gridSize = new dim3(
                    (uint)((width + blockSize.x - 1) / blockSize.x),
                    (uint)((height + blockSize.y - 1) / blockSize.y),
                    1);

                // Launch the kernel with parameters using modern syntax
                shaderKernel.BlockDimensions = blockSize;
                shaderKernel.GridDimensions = gridSize;
                shaderKernel.Run(
                    outputBuffer.DevicePointer,
                    width,
                    height,
                    time,
                    mouseX,
                    mouseY,
                    iconInfoBuffer.DevicePointer,
                    desktopIcons.Length
                );

                // Copy data from device to bitmap
                unsafe
                {
                    outputBuffer.CopyToHost((IntPtr)outputBitmapData);
                }

                // Force form to redraw
                this.Invalidate();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Render error: {ex.Message}");
            }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            if (outputBitmap != null)
            {
                e.Graphics.DrawImage(outputBitmap, 0, 0);
            }

            base.OnPaint(e);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                renderTimer?.Stop();
                renderTimer?.Dispose();

                // Unlock bitmap if needed
                if (outputBitmap != null)
                {
                    outputBitmap.Dispose();
                }

                // Clean up CUDA resources
                iconInfoBuffer?.Dispose();
                outputBuffer?.Dispose();
                cudaContext?.Dispose();
            }

            base.Dispose(disposing);
        }
    }
}