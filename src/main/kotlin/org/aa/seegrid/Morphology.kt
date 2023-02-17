package org.aa.seegrid

import nu.pattern.OpenCV
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc.*
import kotlin.system.exitProcess


internal class Morphology {
    fun run() {
        // Load the image
        val src = Imgcodecs.imread("src/main/resources/images/sudoku sample.jpg")

        // Check if image is loaded fine
        if (src.empty()) {
            println("Error opening image")
            exitProcess(-1)
        }

        // Show source image
        HighGui.imshow("src", src)
        //! [load_image]

        //! [gray]
        // Transform source image to gray if it is not already
        var gray = Mat()
        if (src.channels() == 3) {
            cvtColor(src, gray, COLOR_BGR2GRAY)
        } else {
            gray = src
        }

        // Show gray image
        showWaitDestroy("gray", gray)
        //! [gray]

        //! [bin]
        // Apply adaptiveThreshold at the bitwise_not of gray
        val bw = Mat()
        Core.bitwise_not(gray, gray)
        adaptiveThreshold(gray, bw, 255.0, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2.0)

        // Show binary image
        showWaitDestroy("binary", bw)
        //! [bin]

        //! [init]
        // Create the images that will use to extract the horizontal and vertical lines
        val horizontal = bw.clone()
        val vertical = bw.clone()
        //! [init]

        //! [horiz]
        // Specify size on horizontal axis
        val horizontalSize = horizontal.cols() / 2

        // Create structure element for extracting horizontal lines through morphology operations
        val horizontalStructure = getStructuringElement(
            MORPH_RECT, Size(
                horizontalSize.toDouble(),
                1.0
            )
        )

        // Apply morphology operations
        erode(horizontal, horizontal, horizontalStructure)
        dilate(horizontal, horizontal, horizontalStructure)

        // Show extracted horizontal lines
        showWaitDestroy("horizontal", horizontal)
        //! [horiz]

        //! [vert]
        // Specify size on vertical axis
        val verticalSize = vertical.rows() / 2

        // Create structure element for extracting vertical lines through morphology operations
        val verticalStructure = getStructuringElement(MORPH_RECT, Size(1.0, verticalSize.toDouble()))

        // Apply morphology operations
        erode(vertical, vertical, verticalStructure)
        dilate(vertical, vertical, verticalStructure)

        // Show extracted vertical lines
        showWaitDestroy("vertical", vertical)
        //! [vert]

        //! [smooth]
        // Inverse vertical image
        Core.bitwise_not(vertical, vertical)
        showWaitDestroy("vertical_bit", vertical)

        // Extract edges and smooth image according to the logic
        // 1. extract edges
        // 2. dilate(edges)
        // 3. src.copyTo(smooth)
        // 4. blur smooth img
        // 5. smooth.copyTo(src, edges)

        // Step 1
        val edges = Mat()
        adaptiveThreshold(
            vertical,
            edges,
            255.0,
            ADAPTIVE_THRESH_MEAN_C,
            THRESH_BINARY,
            3,
            -2.0
        )
        showWaitDestroy("edges", edges)

        // Step 2
        val kernel = Mat.ones(2, 2, CvType.CV_8UC1)
        dilate(edges, edges, kernel)
        showWaitDestroy("dilate", edges)

        // Step 3
        val smooth = Mat()
        vertical.copyTo(smooth)

        // Step 4
        blur(smooth, smooth, Size(2.0, 2.0))

        // Step 5
        smooth.copyTo(vertical, edges)

        // Show final result
        showWaitDestroy("smooth - final", vertical)
        //! [smooth]
        exitProcess(0)
    }

    private fun showWaitDestroy(winname: String, img: Mat) {
        HighGui.imshow(winname, img)
        HighGui.moveWindow(winname, 500, 0)
        HighGui.waitKey(0)
        HighGui.destroyWindow(winname)
    }
}

object MorphologyMain {
    @JvmStatic
    fun main(args: Array<String>) {
        // Load the native library.
        OpenCV.loadShared()
        Morphology().run()
    }
}