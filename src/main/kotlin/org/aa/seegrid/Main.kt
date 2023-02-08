package org.aa.seegrid

import javafx.animation.AnimationTimer
import javafx.application.Application
import javafx.scene.Scene
import javafx.scene.image.Image
import javafx.scene.image.ImageView
import javafx.scene.layout.HBox
import javafx.stage.Stage
import nu.pattern.OpenCV
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.opencv.objdetect.Objdetect
import org.opencv.videoio.VideoCapture
import java.io.ByteArrayInputStream
import kotlin.math.roundToInt


fun main() {
    OpenCV.loadShared()
    Application.launch(FxApp::class.java)
}

class FxApp : Application() {
    private var camera: VideoCapture = VideoCapture(0)

    override fun start(primaryStage: Stage?) {
        val cameraView = ImageView()
        val workingView = ImageView()
        val hbox = HBox(cameraView, workingView)
        val scene = Scene(hbox)
        val stage = Stage()
        stage.setScene(scene)
        stage.show()

        object : AnimationTimer() {
            override fun handle(l: Long) {
                val camShot = captureCamera()
                cameraView.setImage(camShot.showFaces().toFxImage())
                workingView.setImage(camShot.toGrayScale().toFxImage())
            }
        }.start()
    }

    private fun captureCamera(): Mat {
        val mat = Mat()
        camera.read(mat)
        return mat
    }
}

private fun Mat.showFaces(): Mat {
    return drawRectangles(detectFaces())
}

private fun Mat.detectFaces(): MatOfRect {
    val cascadeClassifier = CascadeClassifier()
    val minFaceSize = (this.rows() * 0.1f).roundToInt().toDouble()
    cascadeClassifier.load("src/main/resources/haarcascade_frontalface_alt.xml")
    val detectedFaces = MatOfRect()
    cascadeClassifier.detectMultiScale(
        this,
        detectedFaces,
        1.1,
        3,
        Objdetect.CASCADE_SCALE_IMAGE,
        Size(minFaceSize, minFaceSize),
        Size()
    )
    return detectedFaces
}

private fun Mat.drawRectangles(rectangles: MatOfRect): Mat {
    val enhanced = clone()
    for (rectangle in rectangles.toArray())
        Imgproc.rectangle(enhanced, rectangle.tl(), rectangle.br(), Scalar(0.0, 0.0, 255.0), 3)
    return enhanced
}

private fun Mat.toGrayScale(): Mat =
    transform { source, destination ->
        Imgproc.cvtColor(
            source,
            destination,
            Imgproc.COLOR_RGB2GRAY
        )
    }.transform { source, destination ->
        Imgproc.adaptiveThreshold(
            source,
            destination,
            255.0,
            Imgproc.ADAPTIVE_THRESH_MEAN_C,
            Imgproc.THRESH_BINARY,
            31,
            10.0
        )
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
