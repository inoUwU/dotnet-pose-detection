using Microsoft.ML;
using Microsoft.ML.Data;
using System.Windows.Media;

namespace dotnet_pose_detection;

public class OnnxModelScorer
{
    private readonly string imagesFolder;
    private readonly string modelLocation;
    private readonly MLContext mlContext;

    public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
    {
        this.imagesFolder = imagesFolder;
        this.modelLocation = modelLocation;
        this.mlContext = mlContext;
    }

    public struct ImageNetSettings
    {
        public const int imageHeight = 416;
        public const int imageWidth = 416;
    }

    public struct TinyYoloModelSettings
    {
        // for checking Tiny yolo2 Model input and  output  parameter names,
        //you can use tools like Netron, 
        // which is installed by Visual Studio AI Tools

        // input tensor name
        public const string ModelInput = "image";

        // output tensor name
        public const string ModelOutput = "grid";
    }

    private ITransformer LoadModel(string modelLocation)
    {
        // Create IDataView from empty list to obtain input data schema
        var data = mlContext.Data.LoadFromEnumerable(new List<ImageMetadata>());

        // Define scoring pipeline
        var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: this.imagesFolder, inputColumnName: "ImagePath")
                        .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "image"))
                        .Append(mlContext.Transforms.ExtractPixels(outputColumnName: TinyYoloModelSettings.ModelInput, orderOfExtraction: Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator.ColorsOrder.ARGB))
                        .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput }, inputColumnNames: new[] { TinyYoloModelSettings.ModelInput }));

        // Fit scoring pipeline
        var model = pipeline.Fit(data);

        return model;
    }

    private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
    {
        IDataView scoredData = model.Transform(testData);
        IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);
        return probabilities;
    }

    public IEnumerable<float[]> Score(IDataView data)
    {
        var model = LoadModel(modelLocation);
        return PredictDataUsingModel(data, model);
    }
}
