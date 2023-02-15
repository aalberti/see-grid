package org.aa.seegrid

import javafx.application.Application
import javafx.scene.Scene
import javafx.scene.image.Image
import javafx.scene.image.ImageView
import javafx.scene.layout.HBox
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
        initView().setImage(
            loadImage("src/main/resources/images/sudoku sample.jpg")
                .toGrayScale()
                .contours()
                .toFxImage()
        )
    }

    private fun initView(): ImageView {
        val workingView = ImageView()
        val hbox = HBox(workingView)
        val scene = Scene(hbox)
        val stage = Stage()
        stage.setScene(scene)
        stage.show()
        return workingView
    }

    fun loadImage(imagePath: String): Mat {
        return Imgcodecs.imread(imagePath)
    }
}

private fun Mat.toGrayScale(): Mat =
    transform { source, destination ->
        cvtColor(source, destination, COLOR_BGR2GRAY)
        blur(destination, destination, Size(3.0, 3.0))
        Canny(destination, destination, 40.0, 80.0)
    }

private fun Mat.contours(): Mat {
    val contours: List<MatOfPoint> = ArrayList()
    val hierarchy = Mat()
    findContours(this, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE)
    val destination = Mat.zeros(size(), CV_8UC3)
    for (i in contours.indices)
        drawContours(destination, contours, i, Scalar(255.0, 0.0, 255.0), 2, LINE_8, hierarchy, 0, Point())
    return destination
}

private fun Mat.transform(transformer: Mat.(Mat, Mat) -> Unit): Mat {
    val destination = Mat(rows(), cols(), type())
    transformer(this, destination)
    return destination
}

private fun Mat.toFxImage(): Image {
    val bytes = MatOfByte()
    Imgcodecs.imencode(".png", this, bytes)
    val inputStream = ByteArrayInputStream(bytes.toArray())
    return Image(inputStream)
}
