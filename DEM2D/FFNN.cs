using ScottPlot.Plottables;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace DEM2D
{
    public class FFNN : Module<Tensor, Tensor>
    {
        private readonly Linear layer1;
        private readonly Linear layer2;
        private readonly Linear layer3;

        public FFNN(int input_dim, int hidden_dim, int output_dim) : base("FFNN")
        {
            layer1 = Linear(input_dim, hidden_dim);
            layer2 = Linear(hidden_dim, hidden_dim);
            layer3 = Linear(hidden_dim, output_dim);
            RegisterComponents();
          
        }
     
        public override Tensor forward(Tensor input)
        {
            var x = tanh(layer1.forward(input));
            x = tanh(layer2.forward(x));
            x = layer3.forward(x);
            return x;
        }
    }
}
