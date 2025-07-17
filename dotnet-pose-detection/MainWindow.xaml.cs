using System.Drawing;
using System.Drawing.Drawing2D;
using Microsoft.ML;
using System.Windows;
using System.IO;
using System.Windows.Media.Imaging;
using System.Windows.Media;

namespace dotnet_pose_detection;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    string? _ImagePath = null;
    static string assetsRelativePath = @"./assets";
    static string assetsPath = GetAbsolutePath(assetsRelativePath);
    static string modelFilePath = System.IO.Path.Combine(assetsPath, "Model", "tinyyolov2-7.onnx");
    static string imagesFolder = System.IO.Path.Combine(assetsPath, "images");
    static string outputFolder = System.IO.Path.Combine(assetsPath, "images", "output");

    MLContext mlContext = new MLContext();

    public MainWindow()
    {
        InitializeComponent();
        Console.WriteLine(modelFilePath);
    }
    static string GetAbsolutePath(string relativePath)
    {
        FileInfo _dataRoot = new FileInfo(typeof(MainWindow).Assembly.Location);
        string assemblyFolderPath = _dataRoot?.Directory!.FullName!;

        string fullPath = System.IO.Path.Combine(assemblyFolderPath, relativePath);

        return fullPath;
    }

    private void OpenImageBtn_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new Microsoft.Win32.OpenFileDialog
        {
            Title = "画像を選択してください。",
            Filter = "画像ファイル|*.gif;*.png;*.jpeg;*.jpg;*.bmp|すべてのファイル|*.*"
        };

        if (dlg.ShowDialog() == true)
        {
            this._ImagePath = dlg.FileName;
        }

        this.OriginalImage.Source = this._ImagePath == null ? null : new BitmapImage(new Uri(this._ImagePath));
    }

    private void DetectByYoLoBtn_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrEmpty(_ImagePath))
        {
            MessageBox.Show("まず画像を選択してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        try
        {
            Console.WriteLine("物体検出開始...");

            // 選択された画像のフォルダとファイル名を取得
            string selectedImageFolder = Path.GetDirectoryName(_ImagePath)!;
            string selectedImageFileName = Path.GetFileName(_ImagePath);

            Console.WriteLine($"選択された画像: {_ImagePath}");
            Console.WriteLine($"画像フォルダ: {selectedImageFolder}");
            Console.WriteLine($"モデルファイル: {modelFilePath}");

            // 一時的に選択された画像だけのリストを作成
            var selectedImage = new List<ImageMetadata>
            {
                new ImageMetadata
                {
                    ImagePath = _ImagePath,
                    Label = selectedImageFileName
                }
            };

            Console.WriteLine("ImageMetadataリスト作成完了");

            IDataView imageDataView = mlContext.Data.LoadFromEnumerable(selectedImage);
            Console.WriteLine("IDataView作成完了");

            // Create instance of model scorer
            var modelScorer = new OnnxModelScorer(selectedImageFolder, modelFilePath, mlContext);
            Console.WriteLine("OnnxModelScorer作成完了");

            // Use model to score data
            Console.WriteLine("モデル推論開始...");
            IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);
            Console.WriteLine("モデル推論完了");

            // Post-process model output
            YoloOutputParser parser = new YoloOutputParser();

            var boundingBoxes = probabilities
                .Select(probability => parser.ParseOutputs(probability))
                .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

            // Draw bounding boxes for detected objects
            var detectedObjects = boundingBoxes.First();
            string outputPath = Path.Combine(Path.GetDirectoryName(_ImagePath)!, "output");

            DrawBoundingBox(selectedImageFolder, outputPath, selectedImageFileName, detectedObjects);
            LogDetectedObjects(selectedImageFileName, detectedObjects);

            // 処理結果の画像を表示
            string outputImagePath = Path.Combine(outputPath, selectedImageFileName);
            if (File.Exists(outputImagePath))
            {
                this.DetectedImage.Source = new BitmapImage(new Uri(outputImagePath));
            }

            MessageBox.Show($"物体検出が完了しました。{detectedObjects.Count}個のオブジェクトが検出されました。",
                          "検出完了", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"エラーが発生しました: {ex.Message}\n\nスタックトレース:\n{ex.StackTrace}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            Console.WriteLine($"エラー詳細: {ex}");
        }
    }

    private void DetectByOpenCvBtn_Click(object sender, RoutedEventArgs e)
    {

    }

    private static void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
    {
        var image = System.Drawing.Image.FromFile(Path.Combine(inputImageLocation, imageName));

        var originalImageHeight = image.Height;
        var originalImageWidth = image.Width;

        foreach (var box in filteredBoundingBoxes)
        {
            // Get Bounding Box Dimensions
            var x = (uint)Math.Max(box.Dimensions.X, 0);
            var y = (uint)Math.Max(box.Dimensions.Y, 0);
            var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
            var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

            // Resize To Image
            x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
            y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
            width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
            height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

            // Bounding Box Text
            string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

            using (Graphics thumbnailGraphic = Graphics.FromImage(image))
            {
                thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                // Define Text Options
                System.Drawing.Font drawFont = new System.Drawing.Font("Arial", 12, System.Drawing.FontStyle.Bold);
                SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                SolidBrush fontBrush = new SolidBrush(System.Drawing.Color.Black);
                System.Drawing.Point atPoint = new System.Drawing.Point((int)x, (int)y - (int)size.Height - 1);

                // Define BoundingBox options
                System.Drawing.Pen pen = new System.Drawing.Pen(box.BoxColor, 3.2f);
                SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                // Draw text on image 
                thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)y - (int)size.Height - 1, (int)size.Width, (int)size.Height);
                thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                // Draw bounding box on image
                thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
            }
        }

        if (!Directory.Exists(outputImageLocation))
        {
            Directory.CreateDirectory(outputImageLocation);
        }

        image.Save(Path.Combine(outputImageLocation, imageName));
    }

    private static void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
    {
        Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

        foreach (var box in boundingBoxes)
        {
            Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
        }

        Console.WriteLine("");
    }
}