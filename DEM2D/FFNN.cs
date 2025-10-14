using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace DEM2D
{
    public class FFNN : Module<Tensor, Tensor>
    {
        private readonly List<Module<Tensor, Tensor>> layers;

        public FFNN(params int[] layer_param) : base("FFNN")
        {
            layers = [];
            for (int i = 0; i < layer_param.Length - 1; i++)
            {
                var layer = Linear(layer_param[i], layer_param[i + 1]);
                layers.Add(layer);
                register_module("layer" + i, layer);

            }

        }
     
        public override Tensor forward(Tensor input)
        {
            for (int i = 0; i < layers.Count - 1; i++)
            {
                input = tanh(layers[i].forward(input));
            }
            input = layers.Last().forward(input);
            return input;
        }
    }
}
