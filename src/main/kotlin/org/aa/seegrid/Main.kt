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
import org.opencv.ml.KNearest
import org.opencv.ml.Ml.ROW_SAMPLE
import org.opencv.objdetect.HOGDescriptor
import org.opencv.utils.Converters.*
import java.io.ByteArrayInputStream
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.extension


fun main() {
    OpenCV.loadShared()
    Application.launch(FxApp::class.java)
}

class FxApp : Application() {
    override fun start(primaryStage: Stage?) {
        val original = loadImage("src/main/resources/images/voisimage droite.jpg")
            .resize(700)
        val (imageWithContours, contours) = original.toGrayScale().threshold().contours()
        val (_, biggestContour) = imageWithContours.biggestContour(contours)
        val trapezoidContour = original.trapezoidContour(biggestContour)
        val deskewedImage = original.toRectangle(trapezoidContour.contour)
        val cleanedUpDeskewed = deskewedImage.toGrayScale().threshold()
        val deskewedContours = cleanedUpDeskewed.contours().image
        val verticals = cleanedUpDeskewed.verticalLines().contours().image
        val horizontals = cleanedUpDeskewed.horizontalLines().contours().image
        val numbers = cleanedUpDeskewed.numberCandidates()

        trainModel()
        initView(original, deskewedImage, cleanedUpDeskewed, deskewedContours, numbers, verticals, horizontals)
    }

    private fun initView(vararg images: Mat) {
        val pane = FlowPane()
        pane.prefWrapLength = 1900.0
        pane.children.addAll(images.map { it.toFxImage().toImageView() })

        val stage = Stage()
        stage.scene = Scene(ScrollPane(pane))
        stage.width = 1900.0
        stage.height = 1000.0
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
}

private fun loadImage(@Suppress("SameParameterValue") imagePath: String): Mat {
    return Imgcodecs.imread(imagePath)
}

private fun Mat.resize(maxWidth: Int): Mat {
    val ratio: Int = if (size().width > maxWidth) (size().width / maxWidth).toInt() else 1
    return resize(size().width / ratio, size().height / ratio)
}

private fun Mat.resize(width: Double, height: Double): Mat {
    val size = Size(width, height)
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
    val transform = getPerspectiveTransform(trapezoid.toFloats(), rectangle.toFloats())
    val destination = Mat(size(), type())
    copyTo(destination)
    warpPerspective(this, destination, transform, destination.size())
    return destination
}

private fun Mat.verticalLines(): Mat {
    val verticalSize: Int = rows() / 20
    val structureSize = Size(1.0, verticalSize.toDouble())

    return structure(structureSize)
}

private fun Mat.horizontalLines(): Mat {
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

private fun MatOfPoint.toFloats(): MatOfPoint2f {
    val destination = MatOfPoint2f()
    convertTo(destination, CV_32F)
    return destination
}

private fun MatOfPoint2f.toInts(): MatOfPoint {
    val destination = MatOfPoint()
    convertTo(destination, CV_32S)
    return destination
}

private fun Mat.numberCandidates(): Mat {
    val numbers = contours().contours
        .map { it.toFloats() }
        .filter {
            val boundingRect = minAreaRect(it).boundingRect()
            boundingRect.width in 4..20 && boundingRect.height in 12..20
        }.map { it.toInts() }
    val destination = Mat.zeros(size(), CV_8UC3)
    for (i in numbers.indices)
        drawContours(destination, numbers, i, Scalar(255.0, 0.0, 255.0), 1, LINE_8, Mat(), 0, Point())
    return destination
}

private fun trainModel() {
    val labeledImages = Files.walk(Path.of("C:\\Users\\Antoine Alberti\\Pictures\\numbers"))
        .filter { it.extension == "png" }
        .map { Pair(it.parent.fileName.toString().toInt(), it) }
        .toList()
        .shuffled()
    val (trainingLabeledImages, testLabeledImages) = labeledImages.chunked(labeledImages.size * 9 / 10)
    val (trainingFeatures, trainingLabels) = mlData(trainingLabeledImages)

    val knn = KNearest.create()
    knn.defaultK = 5
    knn.train(trainingFeatures, ROW_SAMPLE, trainingLabels)

    val (testFeatures, testLabels, testImagePaths) = mlData(testLabeledImages)
    for (i in 0 until testFeatures.rows()) {
        val nearest = knn.findNearest(testFeatures.row(i), 5, Mat()).toDouble()
        val expected = testLabels.get(i, 0)[0]
        println("${if (nearest == expected) "-" else "X"} got $nearest expected $expected for ${testImagePaths[i]}")
    }
}

private fun mlData(labeledImages: List<Pair<Int, Path>>): Triple<Mat, Mat, List<Path>> {
    val images = labeledImages.asSequence()
        .map { it.second }
        .map { loadImage(it.toString()) }
        .map { it.resize(20.0, 20.0) }
        .map { it.toGrayScale() }
        .toList()
    val features = images.toFeatures()
    val labels = vector_int_to_Mat(
        labeledImages
        .map { it.first }
        .toList()
    ).toFloats()
    return Triple(features, labels, labeledImages.map { it.second })
}

private fun List<Mat>.toFeatures(): Mat {
    val result = Mat()
    this
        .map { it.hogFeatures() }
        .forEach { result.push_back(it) }
    return result
}

private fun Mat.hogFeatures(): Mat {
    val hog = HOGDescriptor(
        Size(20.0, 20.0),
        Size(10.0, 10.0),
        Size(5.0, 5.0),
        Size(5.0, 5.0),
        9
    )

    val descriptor = MatOfFloat()
    hog.compute(this, descriptor)
    return descriptor.columnToRow()
}

private fun Mat.columnToRow():Mat {
    val result = Mat(1, rows(), type())
    for (i in 0 until rows()) result.put(0, i, get(i, 0)[0])
    return result
}

private fun Mat.toFloats(): Mat {
    val destination = Mat(rows(), cols(), CV_32F)
    convertTo(destination, CV_32F)
    return destination
}
