package org.aa.seegrid

import javafx.application.Application
import javafx.scene.Scene
import javafx.scene.control.ScrollPane
import javafx.scene.image.Image
import javafx.scene.image.ImageView
import javafx.scene.layout.FlowPane
import javafx.stage.Stage
import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.core.CvType.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc.*
import java.io.ByteArrayInputStream


fun main() {
    OpenCV.loadShared()
    Application.launch(FxApp::class.java)
}

class FxApp : Application() {
    override fun start(primaryStage: Stage?) {
        val original = loadImage("src/main/resources/images/voisimage droite.jpg")
            .resize()
        val threshold = original.toGrayScale().threshold()
        val (imageWithContours, contours) = threshold.contours()
        val (_, biggestContour) = imageWithContours.biggestContour(contours)
        val trapezoidContour = original.trapezoidContour(biggestContour)
        val deskewedImage = original.toRectangle(trapezoidContour.contour)
        val verticals = deskewedImage.toGrayScale().threshold().verticalLines().contours().image
        val horizontals = deskewedImage.toGrayScale().threshold().horizontalLines().contours().image

        initView(original, deskewedImage, verticals, horizontals)
    }

    private fun initView(vararg images: Mat) {
        val pane = FlowPane()
        pane.prefWrapLength = 1900.0
        pane.children.addAll(images.map { it.toFxImage().toImageView() })

        val stage = Stage()
        stage.scene = Scene(ScrollPane(pane))
        stage.show()
    }

    private fun Mat.toFxImage(): Image {
        val bytes = MatOfByte()
        Imgcodecs.imencode(".png", this, bytes)
        val inputStream = ByteArrayInputStream(bytes.toArray())
        return Image(inputStream)
    }

    private fun Image.toImageView(): ImageView {
        val imageView = ImageView()
        imageView.image = this
        return imageView
    }

    private fun loadImage(@Suppress("SameParameterValue") imagePath: String): Mat {
        return Imgcodecs.imread(imagePath)
    }
}

private fun Mat.resize(): Mat {
    val maxWidth = 700
    val ratio: Int = if (size().width > maxWidth) (size().width / maxWidth).toInt() else 1
    val size = Size(size().width / ratio, size().height / ratio)
    val resized = Mat(size, type())
    resize(this, resized, size)
    return resized
}

private fun Mat.toGrayScale(): Mat {
    val destination = Mat(rows(), cols(), type())
    cvtColor(this, destination, COLOR_BGR2GRAY)
    blur(destination, destination, Size(2.0, 2.0))
    return destination
}

private fun Mat.threshold(): Mat {
    val destination = Mat()
    Core.bitwise_not(this, destination)
    adaptiveThreshold(destination, destination, 255.0, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2.0)
    return destination
}

private fun Mat.contours(): Contours {
    val contours: List<MatOfPoint> = ArrayList()
    val hierarchy = Mat()
    findContours(this, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE)
    val destination = Mat.zeros(size(), CV_8UC3)
    for (i in contours.indices)
        drawContours(destination, contours, i, Scalar(255.0, 0.0, 255.0), 1, LINE_8, hierarchy, 0, Point())
    return Contours(destination, contours)
}

private fun Mat.biggestContour(contours: List<MatOfPoint>): Contour {
    val destination = Mat.zeros(size(), CV_8UC3)
    copyTo(destination)
    val biggestContour = contours.maxBy { contourArea(it) }
    drawContours(destination, listOf(biggestContour), -1, Scalar(255.0, 0.0, 0.0), 2)
    return Contour(destination, biggestContour)
}

private fun Mat.trapezoidContour(biggestContour: MatOfPoint): Contour {
    val destination = Mat.zeros(size(), CV_8UC3)
    copyTo(destination)
    val approximatedContour = approximate(biggestContour)
    drawContours(destination, listOf(approximatedContour), -1, Scalar(255.0, 255.0, 255.0), 3)
    return Contour(destination, approximatedContour)
}

private fun approximate(contour: MatOfPoint): MatOfPoint {
    val contour2F = MatOfPoint2f()
    contour.convertTo(contour2F, CV_32FC2)
    val approximated2F = MatOfPoint2f()
    val perimeter = arcLength(contour2F, true)
    approxPolyDP(contour2F, approximated2F, 0.05 * perimeter, true)
    val approximatedContour = MatOfPoint()
    approximated2F.convertTo(approximatedContour, CV_32S)
    return approximatedContour
}

data class Contours(val image: Mat, val contours: List<MatOfPoint>)

data class Contour(val image: Mat, val contour: MatOfPoint)

private fun Mat.toRectangle(trapezoid: MatOfPoint): Mat {
    val rectangle = MatOfPoint(
        Point(10.0, 10.0),
        Point(10.0, rows().toDouble() - 10.0),
        Point(cols().toDouble() - 10.0, rows().toDouble() - 10.0),
        Point(cols().toDouble() - 10.0, 10.0)
    )
    val transform = getPerspectiveTransform(trapezoid.to2f(), rectangle.to2f())
    val destination = Mat(size(), type())
    copyTo(destination)
    warpPerspective(this, destination, transform, destination.size())
    return destination
}

private fun Mat.verticalLines():Mat {
    val verticalSize: Int = rows() / 20
    val structureSize = Size(1.0, verticalSize.toDouble())

    return structure(structureSize)
}
private fun Mat.horizontalLines():Mat {
    val horizontalSize: Int = cols() / 20
    val structureSize = Size(horizontalSize.toDouble(), 1.0)

    return structure(structureSize)
}

private fun Mat.structure(structureSize: Size): Mat {
    val destination = clone()
    val structure = getStructuringElement(MORPH_RECT, structureSize)
    erode(destination, destination, structure, Point(-1.0, -1.0))
    dilate(destination, destination, structure, Point(-1.0, -1.0))
    return destination
}

private fun MatOfPoint.to2f(): MatOfPoint2f {
    val destination = MatOfPoint2f()
    convertTo(destination, CV_32F)
    return destination
}