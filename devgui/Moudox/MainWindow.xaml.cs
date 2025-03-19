using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;
using Windows.Foundation;
using Windows.Foundation.Collections;

namespace Moudox
{
    /// <summary>
    /// An empty window that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainWindow : Window
    {
        public MainWindow()
        {
            this.InitializeComponent();
            this.PointerPressed += MainWindow_PointerPressed;
            this.PointerMoved += MainWindow_PointerMoved;
        }

        private void MainWindow_PointerPressed(object sender, PointerRoutedEventArgs e)
        {
            var pointerPoint = e.GetCurrentPoint(this);
            var position = pointerPoint.Position;
            // Handle mouse click event
            // For example, display the position in a TextBlock
            myTextBlock.Text = $"Mouse Clicked at: {position.X}, {position.Y}";
        }

        private void MainWindow_PointerMoved(object sender, PointerRoutedEventArgs e)
        {
            var pointerPoint = e.GetCurrentPoint(this);
            var position = pointerPoint.Position;
            // Handle mouse move event
            // For example, display the position in a TextBlock
            myTextBlock.Text = $"Mouse Moved to: {position.X}, {position.Y}";
        }

        private void myButton_Click(object sender, RoutedEventArgs e)
        {
            myButton.Content = "Clicked";
        }
    }
}
