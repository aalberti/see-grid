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
        val imageView = ImageView()
        val hbox = HBox(imageView)
        val scene = Scene(hbox)
        val stage = Stage()
        stage.setScene(scene)
        stage.show()

        object : AnimationTimer() {
            override fun handle(l: Long) {
                imageView.setImage(captureCamera())
            }

            private fun captureCamera(): Image {
                val mat = Mat()
                camera.read(mat)
                showFaces(mat)
                return toImage(mat)
            }

            private fun toImage(mat: Mat): Image {
                val bytes = MatOfByte()
                Imgcodecs.imencode(".png", mat, bytes)
                val inputStream = ByteArrayInputStream(bytes.toArray())
                return Image(inputStream)
            }

            private fun showFaces(loadedImage: Mat) {
                val cascadeClassifier = CascadeClassifier()
                val minFaceSize = (loadedImage.rows() * 0.1f).roundToInt().toDouble()
                cascadeClassifier.load("src/main/resources/haarcascade_frontalface_alt.xml")
                val facesDetected = MatOfRect()
                cascadeClassifier.detectMultiScale(
                    loadedImage,
                    facesDetected,
                    1.1,
                    3,
                    Objdetect.CASCADE_SCALE_IMAGE,
                    Size(minFaceSize, minFaceSize),
                    Size()
                )
                val facesArray: Array<Rect> = facesDetected.toArray()
                for (face in facesArray) {
                    Imgproc.rectangle(loadedImage, face.tl(), face.br(), Scalar(0.0, 0.0, 255.0), 3)
                }
            }
        }.start()
    }
}