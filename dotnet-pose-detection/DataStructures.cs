using System.Drawing;
using System.IO;
using Microsoft.ML.Data;

namespace dotnet_pose_detection;

public abstract class DimensionsBase
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Height { get; set; }
    public float Width { get; set; }
}

public class BoundingBoxDimensions : DimensionsBase { }

public class YoloBoundingBox
{
    public BoundingBoxDimensions Dimensions { get; set; } = new BoundingBoxDimensions();

    public string Label { get; set; } = "";

    public float Confidence { get; set; }

    public Color BoxColor { get; set; }

    public RectangleF Rect =>
        new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height);
}

public class ImageMetadata
{
    public string ImagePath = "";
    public string Label = "";

    public static IEnumerable<ImageMetadata> ReadFromFile(string imageFolder)
    {
        return Directory
            .GetFiles(imageFolder)
            .Where(filePath => Path.GetExtension(filePath) != ".md")
            .Select(filePath => new ImageMetadata()
            {
                ImagePath = filePath,
                Label = Path.GetFileName(filePath)
            });
    }
}

public class ImageNetData : ImageMetadata
{
    public static new IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
    {
        return Directory
            .GetFiles(imageFolder)
            .Where(filePath => Path.GetExtension(filePath) != ".md")
            .Select(filePath => new ImageNetData()
            {
                ImagePath = filePath,
                Label = Path.GetFileName(filePath)
            });
    }
}

public class ImageNetPrediction
{
    [ColumnName("grid")]
    public float[] PredictedLabels = new float[0];
}
