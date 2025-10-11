using ScottPlot;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace DEM2D
{
    internal class Program
    {
        static void Main(string[] args)
        {
            set_default_device(CUDA);
            var domain_size = (1.0, 0.2);
            var num_points = (30, 10);
            var E = 1e5;
            var nu = 0.3;
            using var pinn = new FlexibleDeepEnergyPINN(domain_size, num_points, E, nu);
            static Tensor ZeroDisplacement(Tensor x, Tensor y) => zeros_like(x);
            pinn.SetBoundaryCondition(
            [
                (BoundaryConditionKey.Left,(ZeroDisplacement,ZeroDisplacement)),
                (BoundaryConditionKey.Right,(ZeroDisplacement,ZeroDisplacement)),
                (BoundaryConditionKey.Bottom,(null,null)),
                (BoundaryConditionKey.Top,(null,null))
            ]);
            static (Tensor, Tensor) ForceFunction(Tensor x, Tensor y)
            {
                var fx = zeros_like(x);
                var fy = zeros_like(y);
                fy[TensorIndex.Colon, ^1] = -400;
                return (fx, fy);
            }
            pinn.SetForce(ForceFunction);
            pinn.Train(10000);

            var (u, v) = pinn.GetDisplacement();
            var uArr = u.T.to(device: CPU).data<float>().ToArray();
            var vArr = v.T.to(device: CPU).data<float>().ToArray();

            int nx = num_points.Item1;
            int ny = num_points.Item2;
            double width = domain_size.Item1;
            double height = domain_size.Item2;

            double[] xs = [.. Enumerable.Range(0, nx).Select(i => i * width / (nx - 1))];
            double[] ys = [.. Enumerable.Range(0, ny).Select(j => j * height / (ny - 1))];

            double[] flatX = new double[nx * ny];
            double[] flatY = new double[nx * ny];
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                {
                    int idx = j * nx + i;
                    flatX[idx] = xs[i];
                    flatY[idx] = ys[j];
                }

            double[,] dispMag = new double[ny, nx];
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                {
                    int idx = j * nx + i;
                    double du = uArr[idx];
                    double dv = vArr[idx];
                    dispMag[j, i] = Math.Sqrt(du * du + dv * dv);
                }

            var mp = new Multiplot();
            mp.AddPlots(2);
            mp.Layout = new ScottPlot.MultiplotLayouts.Grid(1, 2);
            Plot getPlot(int r, int c) => mp.Subplots.GetPlot(r * 2 + c);

            {
                var plt = getPlot(0, 0);
                double scale = 5; 

                var scatt = plt.Add.ScatterPoints(flatX, flatY);
                scatt.MarkerSize = 2;
                double[] defX = new double[flatX.Length];
                double[] defY = new double[flatY.Length];
                for (int k = 0; k < flatX.Length; k++)
                {
                    defX[k] = flatX[k] + scale * uArr[k];
                    defY[k] = flatY[k] + scale * vArr[k];
                }

                var def = plt.Add.ScatterPoints(defX, defY);
                def.Color = Color.FromSDColor(System.Drawing.Color.Red);

                plt.Title("Deformed Shape (Red) vs Original (Black)");
                plt.XLabel("x");
                plt.YLabel("y");
            }

            {
                var plt = getPlot(0, 1);
                var hm = plt.Add.Heatmap(dispMag);
                hm.FlipVertically = true;
                plt.Add.ColorBar(hm);
                plt.Title("Displacement Magnitude");
                plt.XLabel("x");
                plt.YLabel("y");
            }


            mp.SavePng("pinn_visualization.png", 1600, 1200);
            Console.WriteLine("Saved: pinn_visualization.png");
        }
    }
}
