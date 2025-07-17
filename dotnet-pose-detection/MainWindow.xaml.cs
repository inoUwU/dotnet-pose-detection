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

        // ファイルロックを回避するためのBitmapImage設定
        if (this._ImagePath != null)
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.UriSource = new Uri(this._ImagePath);
            bitmap.EndInit();
            bitmap.Freeze();

            this.OriginalImage.Source = bitmap;
        }
        else
        {
            this.OriginalImage.Source = null;
        }
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

            this.DetectedImage.Source = null; // Clear previous image
            DrawBoundingBox(selectedImageFolder, outputPath, selectedImageFileName, detectedObjects);
            LogDetectedObjects(selectedImageFileName, detectedObjects);

            // 処理結果の画像を表示
            string outputImagePath = Path.Combine(outputPath, selectedImageFileName);
            if (File.Exists(outputImagePath))
            {
                // BitmapImageのキャッシュを無効化して、ファイルロックを回避
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.UriSource = new Uri(outputImagePath);
                bitmap.EndInit();
                bitmap.Freeze(); // UIスレッド外でも使用可能にする

                this.DetectedImage.Source = bitmap;
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
        string inputPath = Path.Combine(inputImageLocation, imageName);
        string outputPath = Path.Combine(outputImageLocation, imageName);

        // Create output directory if it doesn't exist
        if (!Directory.Exists(outputImageLocation))
        {
            Directory.CreateDirectory(outputImageLocation);
        }

        // 既存の出力ファイルが存在する場合は削除（ファイルロック対策）
        if (File.Exists(outputPath))
        {
            try
            {
                File.Delete(outputPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ファイル削除エラー: {ex.Message}");
                // ファイルがロックされている場合は、異なる名前で保存
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string fileName = Path.GetFileNameWithoutExtension(imageName);
                string extension = Path.GetExtension(imageName);
                outputPath = Path.Combine(outputImageLocation, $"{fileName}_{timestamp}{extension}");
            }
        }

        // 画像を読み込み、コピーを作成してから編集
        using (var originalImage = System.Drawing.Image.FromFile(inputPath))
        {
            // 元画像のコピーを作成
            using (var image = new Bitmap(originalImage.Width, originalImage.Height))
            {
                using (var graphics = Graphics.FromImage(image))
                {
                    // 元画像をコピー
                    graphics.DrawImage(originalImage, 0, 0, originalImage.Width, originalImage.Height);

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

                        graphics.CompositingQuality = CompositingQuality.HighQuality;
                        graphics.SmoothingMode = SmoothingMode.HighQuality;
                        graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;

                        // Define Text Options
                        using (var drawFont = new System.Drawing.Font("Arial", 12, System.Drawing.FontStyle.Bold))
                        {
                            SizeF size = graphics.MeasureString(text, drawFont);
                            using (var fontBrush = new SolidBrush(System.Drawing.Color.Black))
                            using (var colorBrush = new SolidBrush(box.BoxColor))
                            using (var pen = new System.Drawing.Pen(box.BoxColor, 3.2f))
                            {
                                System.Drawing.Point atPoint = new System.Drawing.Point((int)x, (int)y - (int)size.Height - 1);

                                // Draw text on image 
                                graphics.FillRectangle(colorBrush, (int)x, (int)y - (int)size.Height - 1, (int)size.Width, (int)size.Height);
                                graphics.DrawString(text, drawFont, fontBrush, atPoint);

                                // Draw bounding box on image
                                graphics.DrawRectangle(pen, x, y, width, height);
                            }
                        }
                    }
                }

                // 画像を保存
                try
                {
                    image.Save(outputPath);
                    Console.WriteLine($"画像を保存しました: {outputPath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"画像保存エラー: {ex.Message}");
                    throw;
                }
            }
        }
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