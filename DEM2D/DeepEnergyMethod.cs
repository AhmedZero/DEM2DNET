using ScottPlot.Plottables;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.nn;

namespace DEM2D
{
    public class DeepEnergyMethod : IDisposable
    {
        private readonly DisposeScope disposeScope; 
        public DeepEnergyMethod((double Length,double Height) domain_size, (int Length, int Height) num_points,double E,double nu)
        {
            disposeScope = NewDisposeScope();
            Domain_Size = domain_size;
            Num_Points = num_points;
            this.E = E;
            Nu = nu;
            x = linspace(0, domain_size.Length, num_points.Length, requires_grad: true);
            y = linspace(0, domain_size.Height, num_points.Height, requires_grad: true);
            MeshGrid = meshgrid([x, y]);
            xy = stack([MeshGrid[0].flatten(), MeshGrid[1].flatten()], dim : 1);
            model = new FFNN([2,128,128,128,2]);
            optimizer = AdamW(model.parameters(), lr: 0.0001);



        }

        public (double Length, double Height) Domain_Size { get; }
        public (int Length, int Height) Num_Points { get; }
        public double E { get; }
        public double Nu { get; }

        private readonly Tensor x;
        private readonly Tensor y;
        private readonly Tensor[] MeshGrid;
        private readonly Tensor xy;
        private readonly FFNN model;
        private readonly Optimizer optimizer;
        private List<(BoundaryConditionKey, (Func<Tensor, Tensor, Tensor>?, Func<Tensor, Tensor, Tensor>?))>? boundary_conditions = null;
        private Func<Tensor, Tensor, (Tensor fx, Tensor fy)>? force = null;

        public void SetBoundaryCondition(List<(BoundaryConditionKey, (Func<Tensor, Tensor, Tensor>?, Func<Tensor, Tensor, Tensor>?))> boundary_conditions)
        {
            this.boundary_conditions = boundary_conditions;
        }
        public void SetForce(Func<Tensor, Tensor, (Tensor fx, Tensor fy)> force)
        {
            this.force = force;
        }
        private Tensor Strain_Energy_Density(Tensor u, Tensor v, Tensor x, Tensor y)
        {
            using var d = NewDisposeScope();

            var outputs = new[] { u, u, v, v };
            var inputs = new[] { x, y, x, y };
            var grad_outputs = new[] { torch.ones_like(u), torch.ones_like(u), torch.ones_like(v), torch.ones_like(v) };

            var gradients = autograd.grad(outputs, inputs, grad_outputs, retain_graph: true, create_graph: true);

            var u_x = gradients[0];
            var u_y = gradients[1];
            var v_x = gradients[2];
            var v_y = gradients[3];

            var epsilon_xx = u_x;
            var epsilon_yy = v_y;
            var epsilon_xy = 0.5 * (u_y + v_x);
            var sigma_xx = E / (1 - Nu * Nu) * (epsilon_xx + Nu * epsilon_yy);
            var sigma_yy = E / (1 - Nu * Nu) * (Nu * epsilon_xx + epsilon_yy);
            var sigma_xy = E / (2 * (1 + Nu)) * epsilon_xy;
            return (0.5 * (sigma_xx * epsilon_xx + sigma_yy * epsilon_yy + 2 * sigma_xy * epsilon_xy)).MoveToOuterDisposeScope();

        }
        private Tensor InternalEnergy(Tensor u, Tensor v)
        {
            var strain_energy = Strain_Energy_Density(u, v, MeshGrid[0], MeshGrid[1]);
            var E_in = sum(strain_energy) * (Domain_Size.Length / Num_Points.Length) * (Domain_Size.Height / Num_Points.Height);
            return E_in;
        }
        private Tensor ExternalWork(Tensor u,Tensor v)
        {

            if (force is null)
                return 0;

            var (fx, fy) = force(MeshGrid[0], MeshGrid[1]);
            var E_ex = sum(fx * u + fy * v) * (Domain_Size.Length / Num_Points.Length) * (Domain_Size.Height / Num_Points.Height);
            return E_ex;
        }
        private Tensor BoundaryConstraint(Tensor u, Tensor v)
        {
            if (boundary_conditions is null)
                return 0;

            Tensor constraint = 0;

            foreach (var (edge, (u_cond, v_cond)) in boundary_conditions)
            {
                Tensor u_edge, v_edge, x_edge, y_edge;
                switch (edge)
                {
                    case BoundaryConditionKey.Left:
                        u_edge = u[0, TensorIndex.Colon];
                        v_edge = v[0, TensorIndex.Colon];
                        x_edge = MeshGrid[0][0, TensorIndex.Colon];
                        y_edge = MeshGrid[1][0, TensorIndex.Colon];
                        break;
                    case BoundaryConditionKey.Right:
                        u_edge = u[^1, TensorIndex.Colon];
                        v_edge = v[^1, TensorIndex.Colon];
                        x_edge = MeshGrid[0][^1, TensorIndex.Colon];
                        y_edge = MeshGrid[1][^1, TensorIndex.Colon];
                        break;
                    case BoundaryConditionKey.Bottom:
                        u_edge = u[TensorIndex.Colon, 0];
                        v_edge = v[TensorIndex.Colon, 0];
                        x_edge = MeshGrid[0][TensorIndex.Colon, 0];
                        y_edge = MeshGrid[1][TensorIndex.Colon, 0];
                        break;
                    case BoundaryConditionKey.Top:
                        u_edge = u[TensorIndex.Colon, ^1];
                        v_edge = v[TensorIndex.Colon, ^1];
                        x_edge = MeshGrid[0][TensorIndex.Colon, ^1];
                        y_edge = MeshGrid[1][TensorIndex.Colon, ^1];
                        break;
                    default:
                        throw new ArgumentException($"Unknown edge: {edge}");
                }

                if (u_cond != null)
                    constraint += mean((u_edge - u_cond(x_edge, y_edge)).pow(2));

                if (v_cond != null)
                    constraint += mean((v_edge - v_cond(x_edge, y_edge)).pow(2));
            }

            return constraint;
        }
        private Tensor DEMPinnLoss()
        {

            var u_v = model.forward(xy).T;
            var u = u_v[0].reshape([Num_Points.Length, Num_Points.Height]);
            var v = u_v[1].reshape([Num_Points.Length, Num_Points.Height]);
            var E_in = InternalEnergy(u, v);
            var E_ex = ExternalWork(u, v);
            var boundary_constraint = BoundaryConstraint(u, v);
            return E_in - E_ex + 1e5 * boundary_constraint;
        }
        public void Train(int num_epochs)
        {
            for (int epoch = 0; epoch < num_epochs; epoch++)
            {
                using var d = NewDisposeScope();
                var loss = optimizer.step(Closures);
                if (epoch % 100 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}, Loss: {loss.item<float>()}");
                }
            }

        }

        private Tensor Closures()
        {
            optimizer.zero_grad();
            var loss = DEMPinnLoss();
            loss.backward();
            return loss;
        }

        public (Tensor, Tensor) GetDisplacement()
        {
            using var d = NewDisposeScope();
            Tensor? u_v;
            using (_ = no_grad())
            {
                u_v = model.forward(xy).T;

            }
            return (u_v[0].reshape([Num_Points.Length, Num_Points.Height]).MoveToOuterDisposeScope(), u_v[1].reshape([Num_Points.Length, Num_Points.Height]).MoveToOuterDisposeScope());
        }
        public void Dispose()
        {
            disposeScope.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
