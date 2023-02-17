package org.aa.seegrid

import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc.*
import java.awt.BorderLayout
import java.util.*
import javax.swing.*
import kotlin.system.exitProcess

internal class FindContours {
    private val rng = Random(12345)

    init {
        val image = loadImage("src/main/resources/images/voisimage picture.jpg")
        initUI(resize(image))
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

    private fun contours(originalImage: Mat, threshold: Int): Mat {
        val grayImage = Mat()
        cvtColor(originalImage, grayImage, COLOR_BGR2GRAY)
        blur(grayImage, grayImage, Size(3.0, 3.0))
        Canny(grayImage, grayImage, threshold.toDouble(), (threshold * 2).toDouble())
        val contours: List<MatOfPoint> = ArrayList()
        val hierarchy = Mat()
        findContours(grayImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE)

        val contoursImage = Mat.zeros(grayImage.size(), CvType.CV_8UC3)
        for (i in contours.indices) {
            val color = Scalar(rng.nextInt(256).toDouble(), rng.nextInt(256).toDouble(), rng.nextInt(256).toDouble())
            drawContours(contoursImage, contours, i, color, 2, LINE_8, hierarchy, 0, Point())
        }
        val biggestContour = contours.maxBy { contourArea(it) }
        val biggestRectangle = boundingRect(biggestContour)
        rectangle(contoursImage, biggestRectangle, Scalar(255.0, 255.0, 255.0), 3, FILLED)
        return contoursImage
    }

    private fun resize(originalImage: Mat): Mat {
        val originalSize = originalImage.size()
        val ratio: Int = if (originalSize.width > 600) (originalSize.width / 600).toInt() else 1
        val size = Size(originalSize.width / ratio, originalSize.height / ratio)
        val resized = Mat(size, originalImage.type())
        resize(originalImage, resized, size)
        return resized
    }

    private fun initUI(originalImage: Mat) {
        val originalLabel = JLabel(ImageIcon(HighGui.toBufferedImage(originalImage)))
        val blackImg = Mat.zeros(originalImage.size(), CvType.CV_8U)
        val contoursLabel = JLabel(ImageIcon(HighGui.toBufferedImage(blackImg)))
        val panel = JPanel()
        panel.add(originalLabel)
        panel.add(contoursLabel)

        val frame = JFrame("Finding contours in your image demo")
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        val originalThreshold = 100
        frame.contentPane.add(slider(originalThreshold) { threshold ->
            refresh(contours(originalImage, threshold), contoursLabel, frame)
        }, BorderLayout.PAGE_START)
        frame.contentPane.add(panel, BorderLayout.CENTER)
        frame.pack()
        frame.isVisible = true

        refresh(contours(originalImage, originalThreshold), contoursLabel, frame)
    }

    private fun refresh(contours: Mat, contoursLabel: JLabel, frame: JFrame) {
        contoursLabel.icon = ImageIcon(HighGui.toBufferedImage(contours))
        frame.repaint()
    }

    @Suppress("SameParameterValue")
    private fun slider(originalThreshold: Int, onUpdate: (Int) -> Unit): JPanel {
        val sliderPanel = JPanel()
        sliderPanel.layout = BoxLayout(sliderPanel, BoxLayout.PAGE_AXIS)
        sliderPanel.add(JLabel("Canny threshold: "))
        val slider = JSlider(0, 255, originalThreshold)
        slider.majorTickSpacing = 20
        slider.minorTickSpacing = 10
        slider.paintTicks = true
        slider.paintLabels = true
        slider.addChangeListener { e ->
            val source = e.source as JSlider
            onUpdate(source.value)
        }
        sliderPanel.add(slider)
        return sliderPanel
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