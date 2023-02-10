package org.aa.seegrid

import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.LINE_8
import java.awt.BorderLayout
import java.awt.Image
import java.util.*
import javax.swing.*
import kotlin.system.exitProcess

internal class FindContours {
    private var imgContoursLabel: JLabel? = null
    private var threshold = 100
    private val rng = Random(12345)

    init {
        val src = loadImage("src/main/resources/images/sudoku sample.jpg")
        val srcGray = Mat()
        Imgproc.cvtColor(src, srcGray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.blur(srcGray, srcGray, Size(3.0, 3.0))
        val img = HighGui.toBufferedImage(src)

        initUI(img, srcGray)
    }

    @Suppress("SameParameterValue")
    private fun loadImage(filename: String): Mat {
        val src = Imgcodecs.imread(filename)
        if (src.empty()) {
            System.err.println("Cannot read image: $filename")
            exitProcess(0)
        }
        return src
    }

    private fun initUI(originalImage: Image, grayMatrix: Mat) {
        val frame = JFrame("Finding contours in your image demo")
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        frame.contentPane.add(slider { update(frame, grayMatrix) }, BorderLayout.PAGE_START)
        val imgPanel = rawImage(originalImage)
        val blackImg = Mat.zeros(grayMatrix.size(), CvType.CV_8U)
        imgContoursLabel = JLabel(ImageIcon(HighGui.toBufferedImage(blackImg)))
        imgPanel.add(imgContoursLabel)
        frame.contentPane.add(imgPanel, BorderLayout.CENTER)
        frame.pack()
        frame.isVisible = true
        update(frame, grayMatrix)
    }

    private fun rawImage(img: Image): JPanel {
        val imgPanel = JPanel()
        val imgSrcLabel = JLabel(ImageIcon(img))
        imgPanel.add(imgSrcLabel)
        return imgPanel
    }

    private fun slider(onUpdate: () -> Unit): JPanel {
        val sliderPanel = JPanel()
        sliderPanel.layout = BoxLayout(sliderPanel, BoxLayout.PAGE_AXIS)
        sliderPanel.add(JLabel("Canny threshold: "))
        val slider = JSlider(0, MAX_THRESHOLD, threshold)
        slider.majorTickSpacing = 20
        slider.minorTickSpacing = 10
        slider.paintTicks = true
        slider.paintLabels = true
        slider.addChangeListener { e ->
            val source = e.source as JSlider
            threshold = source.value
            onUpdate()
        }
        sliderPanel.add(slider)
        return sliderPanel
    }

    private fun update(frame: JFrame, grayMatrix: Mat) {
        val cannyOutput = Mat()
        Imgproc.Canny(grayMatrix, cannyOutput, threshold.toDouble(), (threshold * 2).toDouble())
        val contours: List<MatOfPoint> = ArrayList()
        val hierarchy = Mat()
        Imgproc.findContours(cannyOutput, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
        val drawing = Mat.zeros(cannyOutput.size(), CvType.CV_8UC3)
        for (i in contours.indices) {
            val color = Scalar(rng.nextInt(256).toDouble(), rng.nextInt(256).toDouble(), rng.nextInt(256).toDouble())
            Imgproc.drawContours(drawing, contours, i, color, 2, LINE_8, hierarchy, 0, Point())
        }
        imgContoursLabel!!.icon = ImageIcon(HighGui.toBufferedImage(drawing))
        frame.repaint()
    }

    companion object {
        private const val MAX_THRESHOLD = 255
    }
}

object FindContoursDemo {
    @JvmStatic
    fun main(args: Array<String>) {
        // Load the native OpenCV library
        OpenCV.loadShared()
        // Schedule a job for the event dispatch thread:
        // creating and showing this application's GUI.
        SwingUtilities.invokeLater { FindContours() }
    }
}