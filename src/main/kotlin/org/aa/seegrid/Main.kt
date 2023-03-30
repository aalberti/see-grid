package org.aa.seegrid

import javafx.application.Application
import javafx.scene.Scene
import javafx.scene.image.Image
import javafx.scene.image.ImageView
import javafx.scene.layout.FlowPane
import javafx.stage.Stage
import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.core.CvType.CV_8UC3
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
        initView(original, { m -> m.toGrayScale() }, { m -> m.threshold() }, { m -> m.contours() })
    }

    private fun initView(original: Mat, vararg operators: (Mat) -> Mat) {
        val pane = FlowPane()
        pane.prefWrapLength = 1500.0
        pane.children.add(original.toFxImage().toImageView())
        val nextImages = operators.runningFoldIndexed(original) { _, previous, operator -> operator(previous) }
        pane.children.addAll(nextImages.map { it.toFxImage().toImageView() })

        val stage = Stage()
        stage.scene = Scene(pane)
        stage.show()
    }

    private fun Image.toImageView(): ImageView {
        val imageView = ImageView()
        imageView.image = this
        return imageView
    }

    private fun loadImage(imagePath: String): Mat {
        return Imgcodecs.imread(imagePath)
    }
}

private fun Mat.resize(): Mat {
    val ratio: Int = if (size().width > 600) (size().width / 600).toInt() else 1
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

private fun Mat.canny(): Mat {
    val destination = Mat(rows(), cols(), type())
    Canny(this, destination, 40.0, 80.0)
    return destination
}

private fun Mat.threshold(): Mat {
    val destination = Mat()
    Core.bitwise_not(this, destination)
    adaptiveThreshold(destination, destination, 255.0, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2.0)
    return destination
}

private fun Mat.contours(): Mat {
    val contours: List<MatOfPoint> = ArrayList()
    val hierarchy = Mat()
    findContours(this, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE)
    val destination = Mat.zeros(size(), CV_8UC3)
    for (i in contours.indices)
        drawContours(destination, contours, i, Scalar(255.0, 0.0, 255.0), 1, LINE_8, hierarchy, 0, Point())
    return destination
}

private fun Mat.toFxImage(): Image {
    val bytes = MatOfByte()
    Imgcodecs.imencode(".png", this, bytes)
    val inputStream = ByteArrayInputStream(bytes.toArray())
    return Image(inputStream)
}
