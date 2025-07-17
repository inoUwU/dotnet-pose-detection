using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace dotnet_pose_detection;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    string? _ImagePath = null;

    public MainWindow()
    {
        InitializeComponent();
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

    }

    private void DetectByOpenCvBtn_Click(object sender, RoutedEventArgs e)
    {

    }
}